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
    bat, f_m, h, w = inp.shape
    o_h = int((h + 2 * pad - p_h) / stride + 1)
    o_w = int((w + 2 * pad - p_w) / stride + 1)

    inp_reshaped = inp.reshape(-1, 1, h, w)
    inp_col = img_to_col(inp_reshaped, p_h, p_w, o_h, o_w, pad, stride)

    max_indices = np.argmax(inp_col, axis=0)
    max_vals = inp_col[max_indices, np.arange(inp_col.shape[1])]
    max_reshape = max_vals.reshape(o_h, o_w, -1, f_m)
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


def batch_normalize(inp, gamma, beta, epsilon=1e-8):
    """
    H' = (H - m) / s
    g*H' + b
    g * (H - m) / s + b
    g/s * H - gm/s + b
    g/s * H + b - gm/s
    """
    #mean, variance = tf.nn.moments(inp, axes=[0, 2, 3])

    N, D, h, w = inp.shape
    mean = np.mean(inp, axis=(0, 2, 3))
    variance = np.mean((inp - mean[0]) * (inp - mean[0]))

    std = np.sqrt(variance + epsilon)
    norm_inp = gamma / std * inp + (beta - gamma * mean[0] / std)
    return norm_inp #, mean, std


"""******************************************
        ACTIVATION FUNCTIONS
******************************************"""

def relu(inp):
    return np.maximum(inp, 0)


def leaky_relu(inp, alpha=0.01):
    return np.maximum(inp, inp*alpha)


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
    return 1 / 2 * (tf.square(inp - label))


"""******************************************
        BACK PROP OPERATIONS
******************************************"""

def convole_backprop(inp, weights, delta_out, pad, stride):
    fm_h = int((inp.shape[2] + 2 * pad - weights.shape[2]) / stride + 1)
    fm_w = int((inp.shape[3] + 2 * pad - weights.shape[3]) / stride + 1)
    weights2d = weights.reshape(weights.shape[0], -1)
    inp2d = img_to_col(inp, weights.shape[2], weights.shape[3], fm_h, fm_w, pad, stride)

    dO = delta_out.transpose(1, 2, 3, 0)
    dO = dO.reshape(weights.shape[0], -1)

    dx = tf.matmul(weights2d.T, dO)
    dx = col_to_img(dx, inp.shape, weights.shape[2], weights.shape[3], fm_h, fm_w, pad, stride)

    dw = tf.matmul(dO, inp2d)
    dw = dw.reshape(weights.shape[0], weights.shape[2], weights.shape[3], weights.shape[1])
    dw = dw.transpose(0, 3, 1, 2)
    return dw, dx

def pool_backprop(inp, p_h, p_w, stride, pad, error):
    bat, f_m, h, w = inp.shape
    o_h = int((h + 2 * pad - p_h) / stride + 1)
    o_w = int((w + 2 * pad - p_w) / stride + 1)
    bat, e_d, e_h, e_w = error.shape

    inp_reshaped = inp.reshape(-1, 1, h, w)
    inp_col = img_to_col(inp_reshaped, p_h, p_w, o_h, o_w, pad, stride)

    err = error.transpose(2,3,1,0).reshape(-1)

    max_indices = np.argmax(inp_col, axis=0)
    back = np.zeros_like(inp_col)
    np.add.at(back, (max_indices, np.arange(inp_col.shape[1])), err)
    back = col_to_img(back, inp_reshaped.shape, p_h, p_w, e_h, e_w, pad, stride)
    return back

def batch_norm_backprop(delta_out, cache):
    N, D = delta_out.shape
    mean, inv_var, x_hat, gamma = cache

    # intermediate partial derivatives
    dxhat = delta_out * gamma

    # final partial derivatives
    dx = (1. / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=0) - x_hat*np.sum(dxhat*x_hat, axis=0))
    db = np.sum(delta_out, axis=0)
    dg = np.sum(x_hat*delta_out, axis=0)

    return dx, dg, db


"""******************************************
        DERIVATIVES
******************************************"""

def relu_prime(inp):
    return tf.where(inp < 0, 0, 1)


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

    img = np.zeros(inp_shape, dtype=float)
    cols_reshaped = col.reshape(i_c*f_h*f_w, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(img, (slice(None), z, y, x), cols_reshaped)

    if pad == 0:
        return img
    return img[:, :, pad:-pad, pad:-pad]


def intersection_over_union(box1, box2, img_width, img_height):
    I = _find_intersection(box1, box2, img_width, img_height)
    U = (box1[3] * box1[4] + box2[3] * box2[4]) - I
    return I / U


def _find_intersection(box1, box2, width, height):
    TL1, TR1, BL1, BR1 = _find_corners(box1)
    b1 = np.zeros((height, width))
    b1x1 = TL1[1]
    b1x2 = TR1[1]
    b1y1 = TL1[0]
    b1y2 = BL1[0]
    b1[b1y1:b1y2, b1x1:b1x2] = 1

    TL2, TR2, BL2, BR2 = _find_corners(box2)
    b2 = np.zeros((height, width))
    b2x1 = TL2[1]
    b2x2 = TR2[1]
    b2y1 = TL2[0]
    b2y2 = BL2[0]
    b2[b2y1:b2y2, b2x1:b2x2] = 1

    return np.sum(b1 * b2)


def _find_corners(box, index):
    bb_h = np.floor(box[3] * IMG_HEIGHT)
    bb_w = np.floor(box[4] * IMG_WIDTH)
    y_mid = box[1] * CELL_HEIGHT
    x_mid = box[2] * CELL_WIDTH

    grid_x = index % GRID_WIDTH
    grid_y = (index - grid_x) / GRID_WIDTH

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
    predictions = np.array([], dtype=int)
    i = 0
    while i < bounding_boxes.shape[0]:
        if bounding_boxes[i][0] < conf_threshold:
            bounding_boxes[i][0] = 0
            i = i + 1
        else:
            p = np.argmax(bounding_boxes.T[0], axis=0)
            predictions = np.append(predictions, p)
            bounding_boxes[predictions[-1]][0] = 0
            for b in bounding_boxes:
                if b[0] != 0:
                    overlap = intersection_over_union(b, bounding_boxes[predictions[-1]], IMG_HEIGHT, IMG_WIDTH)
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
    std = 1/sqrt(np.sum(shape))
    weights = np.random.randn(shape) / std
    return weights


def zero_pad(inp, pad):
    #return tf.pad(inp, [[pad, pad], [pad, pad]], mode='CONSTANT')
    return np.pad(inp, ((pad, pad), (pad, pad)), 'constant', constant_values=0)


def grad_check(weights, grads):
    """
    W = weights
    num_grads = tf.zeros(weights.shape)
    for each weight
        W[weight] += epsilon
        cost_plus = cost(W)
        W[weight] -= 2 * epsilon
        cost_minus = cost(W)
        num_grads[weight] = (cost_plus - cost_minus) / (2 * epsilon)

    rel_error = (grads - num_grads) / ((grads + num _grads) + (epsilon * epsilon))
    return rel_error
    """