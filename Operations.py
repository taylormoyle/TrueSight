import tensorflow as tf
import numpy as np
from math import sqrt

IMG_HEIGHT = 416
IMG_WIDTH = 416
CELL_HEIGHT = 32
CELL_WIDTH = 32
GRID_HEIGHT = 13
GRID_WIDTH = 13

"""******************************************
        FORWARD PROP OPERATIONS
******************************************"""

def pool(inp, p_h, p_w, stride, pad=0):
    N, D, h, w = inp.shape
    o_h = int((h + 2 * pad - p_h) / stride + 1)
    o_w = int((w + 2 * pad - p_w) / stride + 1)

    inp_reshaped = inp.reshape(N*D, 1, h, w)
    inp_col = img_to_col(inp_reshaped, p_h, p_w, o_h, o_w, pad, stride)

    max_indices = np.argmax(inp_col, axis=0)
    max_vals = inp_col[max_indices, np.arange(inp_col.shape[1])]
    max_reshape = max_vals.reshape(o_h, o_w, N, D)
    return max_reshape.transpose(2, 3, 0, 1)


def convolve(inp, weights, stride=1, pad=0):
    bat, i_c, i_h, i_w = inp.shape
    n_f, n_c, f_h, f_w = weights.shape
    o_h = int((i_h + 2 * pad - f_h) / stride + 1)
    o_w = int((i_w + 2 * pad - f_w) / stride + 1)

    inp_col = img_to_col(inp, f_w, f_h, o_h, o_w, pad, stride)
    w_col = weights.reshape(n_f, -1)

    output = np.matmul(w_col, inp_col)
    output = output.reshape(n_f, o_h, o_w, bat)
    return output.transpose(3, 0, 1, 2)


def batch_normalize(inp, bg, mv, training=False, epsilon=1e-8):
    """
    H' = (H - m) / s
    g*H' + b
    g * (H - m) / s + b
    g/s * H - gm/s + b
    g/s * H + b - gm/s
    """
    #mean, variance = tf.nn.moments(inp, axes=[0, 2, 3])

    N, D, h, w = inp.shape
    beta, gamma = bg

    if training:
        mean = np.mean(inp, axis=(0, 2, 3))
        inp_minus_mean = inp - mean.reshape(1, D, 1, 1)
        variance = np.mean(inp_minus_mean * inp_minus_mean, axis=(0, 2, 3))
        inv_std = 1 / np.sqrt(variance.reshape(1, D, 1, 1) + epsilon)
        x_hat = inp_minus_mean * inv_std
    else:
        mean, variance = mv
        inp_minus_mean = inp - mean.reshape(1, D, 1, 1)
        inv_std = 1 / np.sqrt(variance.reshape(1, D, 1, 1) + epsilon)
        x_hat = inp_minus_mean * inv_std

    norm_inp = gamma.reshape(1, D, 1, 1) * x_hat + beta.reshape(1, D, 1, 1)

    cache = mean, inv_std, x_hat, gamma, epsilon
    mean_var = np.stack((mean, variance))
    return norm_inp, cache, mean_var


def full_conn(inp, weights):
    full_conn = np.matmul(inp, weights)
    return full_conn

"""******************************************
        ACTIVATION FUNCTIONS
******************************************"""

def relu(inp):
    return np.where(inp >= 0, inp, 0)


def leaky_relu(inp, alpha=0.01):
    return np.where(inp >= 0, inp, inp*alpha)


def sigmoid(inp):
    return 1 / (1 + (np.exp(-inp)))


"""******************************************
        COST FUNCTIONS
******************************************"""

def cost_function(inp, label, batch_size, lambd_coord, lambd_noobj):
    inp_reshaped = inp.reshape(169, -1)
    label_reshaped = label.reshape(169, -1)

    xy_cost = 0
    wh_cost = 0
    oc_cost = 0
    nc_cost = 0

    for i, l in zip(inp_reshaped, label_reshaped):
        if l[0]:
            xy_cost += mean_square_error(i[1] - l[1]) + mean_square_error(i[2] - l[2])
            wh_cost += mean_square_error(tf.sqrt(i[3]) - tf.sqrt(l[3])) + mean_square_error(tf.sqrt(i[4]) - tf.sqrt(l[4]))
            oc_cost += logistic_regression(i[0] - l[0])
        else:
            nc_cost += logistic_regression(i[0] - l[0])

    return (lambd_coord * (xy_cost + wh_cost)) + oc_cost + (lambd_noobj * nc_cost) / batch_size


def logistic_regression(inp, label):
    return -(label * tf.log(inp) + ((1 - label) * tf.log(1 - inp)))


def mean_square_error(inp, label):
    return 1 / 2 * (np.square(label - inp))


"""******************************************
        BACK PROP OPERATIONS
******************************************"""

def convolve_backprop(delta_out, inp, weights, pad=0, stride=1):
    fm_h = int((inp.shape[2] + 2 * pad - weights.shape[2]) / stride + 1)
    fm_w = int((inp.shape[3] + 2 * pad - weights.shape[3]) / stride + 1)
    weights2d = weights.reshape(weights.shape[0], -1)
    inp2d = img_to_col(inp, weights.shape[2], weights.shape[3], fm_h, fm_w, pad, stride)

    dO = delta_out.transpose(1, 2, 3, 0)
    dO = dO.reshape(weights.shape[0], -1)

    dx = np.matmul(weights2d.transpose(), dO)
    dx = col_to_img(dx, inp.shape, weights.shape[2], weights.shape[3], fm_h, fm_w, pad, stride)

    dw = np.matmul(dO, inp2d.transpose())
    dw = dw.reshape(weights.shape)
    #dw = dw.transpose(0, 3, 1, 2)
    return dx, dw

def pool_backprop(delta_out, inp, p_h, p_w, stride, pad=0):
    N, D, h, w = inp.shape
    o_h = int((h + 2 * pad - p_h) / stride + 1)
    o_w = int((w + 2 * pad - p_w) / stride + 1)

    inp_reshaped = inp.reshape(-1, 1, h, w)
    inp_col = img_to_col(inp_reshaped, p_h, p_w, o_h, o_w, pad, stride)

    err = delta_out.transpose(2,3,0,1).reshape(-1)

    max_indices = np.argmax(inp_col, axis=0)
    back = np.zeros_like(inp_col)
    back[max_indices, np.arange(inp_col.shape[1])] = err
    back = col_to_img(back, inp_reshaped.shape, p_h, p_w, o_h, o_w, pad, stride)
    dx = back.reshape(inp.shape)
    return dx

def batch_norm_backprop(delta_out, cache):
    N, D, _, _ = delta_out.shape
    mean, inv_std, x_hat, gamma, epsilon = cache

    # intermediate partial derivatives
    dx_hat = delta_out * gamma.reshape(1, D, 1, 1)

    # final partial derivatives
    dx_hat_sum = np.sum(dx_hat, axis=(0, 2, 3))
    dx_hat2 = np.sum(dx_hat*x_hat, axis=(0, 2, 3))
    dx = N*dx_hat - dx_hat_sum.reshape(1, D, 1, 1) - x_hat*dx_hat2.reshape(1, D, 1, 1)
    dx = (1. / N) * inv_std * dx
    db = np.sum(delta_out, axis=(0, 2, 3))
    dg = np.sum(x_hat*delta_out, axis=(0, 2, 3))

    dbg = np.stack((db, dg))
    return dx, dbg

def full_conn_backprop(delta_out, inp, weights):
    dx = np.matmul(delta_out, weights.transpose())
    dw = np.matmul(inp.transpose(), delta_out)
    return dx, dw

"""******************************************
        DERIVATIVES
******************************************"""

def relu_prime(inp):
    return np.where(inp < 0, 0, 1)


def leaky_relu_prime(inp, alpha=0.01):
    return tf.where(inp < 0, alpha, 1)


def sigmoid_prime(inp):
    s = sigmoid(inp)
    return s * (1 - s)


def log_reg_prime(inp, label):
    return (label / inp) + ((1 - label) / (1 - inp))


def mse_prime(inp, label):
    return inp - label


def mse_wh_prime(inp, label):
    return 1 / (tf.sqrt(inp) - tf.sqrt(label))


"""******************************************
        UTILITY
******************************************"""

def get_col_indices(inp_shape, f_h, f_w, o_h, o_w, stride):
    _, i_c, i_h, i_w = inp_shape

    # z axis (channel)
    z = np.repeat(np.arange(i_c), f_w * f_h).reshape(-1, 1)

    # y index
    y0 = np.tile(np.repeat(np.arange(f_h), f_w), i_c)
    y1 = stride * np.repeat(np.arange(o_h), o_w)
    y = y0.reshape(-1, 1) + y1.reshape(1, -1)

    # x index
    x0 = np.tile(np.arange(f_w), f_h * i_c)
    x1 = stride * np.tile(np.arange(o_w), o_h)
    x = x0.reshape(-1, 1) + x1.reshape(1, -1)

    return z, y, x


def img_to_col(inp, f_h, f_w, o_h, o_w, pad, stride):
    z, y, x = get_col_indices(inp.shape, f_h, f_w, o_h, o_w, stride)

    inp_padded = np.pad(inp, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    col = inp_padded[:, z, y, x]
    col = col.transpose(1, 2, 0)
    return col.reshape(f_h * f_w * inp.shape[1], -1)


def col_to_img(col, inp_shape, f_h, f_w, o_h, o_w, pad, stride):
    N, i_c, i_h, i_w = inp_shape
    z, y, x = get_col_indices(inp_shape,f_h, f_w, o_h, o_w, stride)

    h_p, w_p = i_h + 2 * pad, i_w + 2 * pad
    img = np.zeros((N, i_c, h_p, w_p), dtype=float)
    cols_reshaped = col.reshape(i_c*f_h*f_w, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(img, (slice(None), z, y, x), cols_reshaped)

    if pad == 0:
        return img
    return img[:, :, pad:-pad, pad:-pad]


def intersection_over_union(box1, box2, box1_cell, box2_cell, img_width, img_height):
    I = _find_intersection(box1, box2, box1_cell, box2_cell, img_width, img_height)
    U = (box1[3] * box1[4] + box2[3] * box2[4]) - I
    return I / U


def _find_intersection(box1, box2, box1_cell, box2_cell, width, height):
    TL1, TR1, BL1, BR1 = _find_corners(box1, box1_cell)
    b1 = np.zeros((height, width))
    b1x1 = TL1[1]
    b1x2 = TR1[1]
    b1y1 = TL1[0]
    b1y2 = BL1[0]
    b1[b1y1:b1y2, b1x1:b1x2] = 1

    TL2, TR2, BL2, BR2 = _find_corners(box2, box2_cell)
    b2 = np.zeros((height, width))
    b2x1 = TL2[1]
    b2x2 = TR2[1]
    b2y1 = TL2[0]
    b2y2 = BL2[0]
    b2[b2y1:b2y2, b2x1:b2x2] = 1

    return np.sum(b1 * b2)


def _find_corners(box, cell):
    bb_h = np.floor(box[3] * IMG_HEIGHT)
    bb_w = np.floor(box[4] * IMG_WIDTH)
    y_mid = box[1] * CELL_HEIGHT
    x_mid = box[2] * CELL_WIDTH

    grid_x = cell % GRID_WIDTH
    grid_y = (cell - grid_x) / GRID_WIDTH

    x = grid_x * CELL_WIDTH + x_mid
    y = grid_y * CELL_HEIGHT + y_mid

    print(x, y, bb_h, bb_w, x_mid, y_mid, grid_x, grid_y)

    bT = int(y - (bb_w / 2))
    bR = int(x + (bb_h / 2))
    bB = int(y + (bb_w / 2))
    bL = int(x - (bb_h / 2))

    TL = [bT, bL]
    TR = [bT, bR]
    BL = [bB, bL]
    BR = [bB, bR]

    return TL, TR, BL, BR


def non_max_suppression(bounding_boxes, conf_threshold, IoU_threshold):
    bounding_boxes = bounding_boxes.reshape(-1, 5)
    bnd_boxes = bounding_boxes
    predictions = np.array([], dtype=int)
    i = 0
    while i < bnd_boxes.shape[0]:
        if bnd_boxes[i][0] < conf_threshold:
            bnd_boxes[i][0] = 0
            i = i + 1
        else:
            p = np.argmax(bnd_boxes.T[0], axis=0)
            predictions = np.append(predictions, p)
            bnd_boxes[predictions[-1]][0] = 0
            for b in range(bnd_boxes):
                if b[0] != 0:
                    overlap = intersection_over_union(bnd_boxes[b], bnd_boxes[predictions[-1]], b, predictions[-1], IMG_HEIGHT, IMG_WIDTH)
                    if overlap >= IoU_threshold:
                        b[0] = 0
                        i = i + 1
    return bounding_boxes[predictions]


def initialize_weights(shape):
    '''
    std_dev = 1 / (sqrt(tf.cumprod(shape)[-1]))
    weights = tf.random_normal(shape, stddev=std_dev)
    return tf.Variable(weights)
    '''
    if len(shape) == 4:
        f, d, h, w = shape
        w_in = d*h*w
        w_out = f
    else:
        w_in = shape[0]
        w_out = shape[1]
    std = 2 / (w_in + w_out)
    weights = np.random.standard_normal(shape) * std
    return weights


def update_weights(weights, gradients, learning_rate, batch_size):
    return weights - (learning_rate / batch_size * gradients)


def zero_pad(inp, pad):
    #return tf.pad(inp, [[pad, pad], [pad, pad]], mode='CONSTANT')
    return np.pad(inp, ((pad, pad), (pad, pad)), 'constant', constant_values=0)


def normalize(inp, epsilon=1e-8):
    mean = np.mean(inp, axis=(2, 3))
    inp_minus_mean = inp - mean.reshape(-1, 3, 1, 1)
    variance = np.mean(inp_minus_mean * inp_minus_mean, axis=(2, 3)).reshape(-1, 3, 1, 1)
    inv_std = 1 / np.sqrt(variance + epsilon)
    return inp_minus_mean * inv_std
