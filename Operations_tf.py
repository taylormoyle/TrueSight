import tensorflow as tf
import numpy as np
from math import floor
from Operations import img_to_col as im2col

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
    '''
    Performs Max-Pooling -- Req1uires Non-Overlapping, Square Windows
    :param inp: Overlay Matrix
    :param p_h: Pooling Window Height
    :param p_w: Pooling Window Width
    :param stride: Stride of Pooling Window
    :param pad: Output Padding
    :return: Max-Pooled Matrix
    '''
    N, D, H, W = tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(inp)[2], tf.shape(inp)[3]
    inp_reshaped = tf.reshape(inp, [N, D, tf.cast(H/p_h, dtype=tf.int32), p_h, tf.cast(W/p_w, dtype=tf.int32), p_w])
    out = tf.reduce_max(tf.reduce_max(inp_reshaped, axis=3), axis=4)
    return out


# Use this one if you have Rectangular windows
def rect_pool(inp, p_h, p_w, stride, pad=0):
    o_h = tf.cast(((h + 2 * pad - p_h) / stride + 1), dtype=tf.int32)
    o_w = tf.cast(((w + 2 * pad - p_w) / stride + 1), dtype=tf.int32)

    inp_reshaped = tf.transpose(tf.reshape(inp, [N * D, 1, h, w]), perm=[0, 2, 3, 1])
    inp_col = img_to_col(inp_reshaped, p_h, p_w, o_h, o_w, pad, stride)
    inp_col = tf.extract_image_patches(inp_reshaped, ksizes=[1, 2, 2, 1], strides=[1, 2, 2, 1], rates=[1, 1, 1, 1],
                                       padding="VALID")
    inp_col = tf.transpose(tf.reshape(inp_col, [-1, p_h * p_w]), perm=[1, 0])

    max_indices = tf.argmax(inp_col, axis=0)
    max_vals = tf.gather(inp_col, [tf.cast(max_indices, dtype=tf.int32), tf.range(tf.shape(inp_col)[1])])
    max_reshape = tf.reshape(max_vals, [o_h, o_w, N, D])
    tf.transpose(max_reshape, perm=[2, 3, 0, 1])


def convolve(inp, weights, stride=1, pad=0):
    '''
    Performs fast-Convolution using im2col to convert to matrix multiplication
    :param inp: Input Images
    :param weights: Convolution Filters
    :param stride: Stride of Convolution Window
    :param pad: Output Padding
    :return: Convolved Images
    '''
    bat, i_c, i_h, i_w = tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(inp)[2], tf.shape(inp)[3]
    n_f, n_c, f_h, f_w = tf.shape(weights)[0], tf.shape(weights)[1], tf.shape(weights)[2], tf.shape(weights)[3]
    o_h = tf.cast(((i_h + 2 * pad - f_h) / stride + 1), dtype=tf.int32)
    o_w = tf.cast(((i_w + 2 * pad - f_w) / stride + 1), dtype=tf.int32)

    #inp_col = img_to_col(inp, f_w, f_h, o_h, o_w, pad, stride)
    #inp_ = tf.transpose(inp, perm=[0, 2, 3, 1])
    #inp_col = tf.extract_image_patches(inp, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
    #inp_col = tf.transpose(tf.reshape(inp_col, [-1, f_h*f_w*n_c]), perm=[1, 0])

    inp_col = tf.py_func(im2col, [inp, f_w, f_h, o_h, o_w, pad, stride], tf.float32)
    w_col = tf.reshape(weights, [n_f, -1])

    output = tf.matmul(w_col, inp_col)
    output = tf.reshape(output, [n_f, o_h, o_w, bat])
    return tf.transpose(output, perm=[3, 0, 1, 2])


def batch_normalize(inp, beta, gamma, running_mean, running_var, training=False, decay=0.9, epsilon=1e-8):
    '''
    Calculates samples Mean and Variance to normalize the Input
    :param inp: Input Batches
    :param beta: Learnable Paramater to shift Output
    :param gamma: Learnable Paramater to scale Output
    :param running_mean: Population Mean
    :param running_var: Population Variance
    :param training: Toggle Training
    :param decay: How much of the previous Population Mean/Variance to keep
    :param epsilon: Small floating-poing number for numerical stability
    :return: Normalized Input
    '''
    N, D, H, W = tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(inp)[2], tf.shape(inp)[3]

    def is_training():
        mean, var = tf.nn.moments(inp, axes=[0, 2, 3], keep_dims=True)

        train_mean = tf.assign(running_mean,
                               decay * running_mean + (1-decay) * mean)
        train_var = tf.assign(running_var,
                              decay * running_var + (1 - decay) * var)

        return mean, var, train_mean, train_var

    def not_training():
        return running_mean, running_var, running_mean, running_var

    mean, var, train_mean, train_var = tf.cond(tf.equal(training, True),
                                                    is_training, not_training)

    with tf.control_dependencies([train_mean, train_var]):
        norm_inp = gamma * (inp - mean) * (1. / tf.sqrt(var + epsilon)) + beta
        xmu=inv_std=x_hat=0
        cache = (xmu, inv_std, x_hat, gamma)
        return norm_inp, cache, train_mean, train_var


def full_conn(inp, weights, bias):
    ''' Computes Dense Layer Outputs '''

    full_conn = tf.matmul(inp, weights) + bias
    return full_conn


"""******************************************
        ACTIVATION FUNCTIONS
******************************************"""

def relu(inp):
    ''' Calculates ReLU Activation (max(0, inp) element-wise) '''
    zeros = tf.zeros_like(inp)
    return tf.where(tf.greater_equal(inp, zeros), inp, zeros)
    #return leaky_relu(inp)


def leaky_relu(inp, alpha=0.01):
    ''' Calculates Leaky ReLU Activation (max(inp*alpha, inp) element-wise) '''
    zeros = tf.zeros_like(inp)
    return tf.where(tf.greater_equal(inp, zeros), inp, inp*alpha)


def sigmoid(inp):
    return 1 / (1 + (tf.exp(-inp)))


"""******************************************
        COST FUNCTIONS
******************************************"""

def cost_function(inp, label, batch_size, lambd_coord, lambd_noobj):
    '''
    Calculates Cost Analysis based on YOLO's Cost Function, used for multi-objhect detection
    :param inp: Output of the Neural Network Graph
    :param label: Ground-Truth Label
    :param batch_size: Size of Batch
    :param lambd_coord: Amount to weigh cost of cells responsible for detecting objects
    :param lambd_noobj: Amount to weigh cost of cells NOT responsible for detecting objects
    :return: Loss
    '''
    inp_reshaped = tf.reshape(inp, [169, -1])
    label_reshaped = tf.reshape(label, [169, -1])

    xy_cost = 0
    wh_cost = 0
    oc_cost = 0
    nc_cost = 0

    for i, l in zip(inp_reshaped, label_reshaped):
        if l[0]:
            xy_cost = xy_cost + mean_square_error(i[1], l[1]) + mean_square_error(i[2], l[2])
            wh_cost = wh_cost + mean_square_error(tf.sqrt(i[3]), tf.sqrt(l[3])) + mean_square_error(tf.sqrt(i[4]), tf.sqrt(l[4]))
            oc_cost = oc_cost + logistic_regression(i[0], l[0])
        else:
            nc_cost = nc_cost + logistic_regression(i[0], l[0])

    return (lambd_coord * (xy_cost + wh_cost)) + oc_cost + (lambd_noobj * nc_cost) / batch_size


def logistic_regression(inp, label):
    return -(label * tf.log(inp) + ((1 - label) * tf.log(1 - inp)))


def mean_square_error(inp, label):
    return (tf.square(tf.cast(label, dtype=tf.float32) - inp))


"""******************************************
        BACK PROP OPERATIONS
******************************************"""

def convolve_backprop(delta_out, inp, weights, pad=0, stride=1):
    N, D, h, w = tf.shape(inp)[0], tf.shape(inp)[1], tf.shape(inp)[2], tf.shape(inp)[3]
    n_fm, n_c, h_w, w_w = tf.shape(weights)[0], tf.shape(weights)[1], tf.shape(weights)[2], tf.shape(weights)[3]

    fm_h = tf.constant(((h + 2 * pad - h_w) / stride + 1), dtype=tf.int32)
    fm_w = tf.constant(((w + 2 * pad - w_w) / stride + 1), dtype=tf.int32)
    weights2d = tf.reshape(weights, [n_fm, -1])
    #inp2d = img_to_col(inp, h_w, w_w, fm_h, fm_w, pad, stride)
    inp2d = tf.py_func(im2col, [inp, h_w, w_w, fm_h, fm_w, pad, stride], tf.float32)

    dO = tf.transpose(delta_out, perm=[1, 2, 3, 0])
    dO = tf.reshape(dO, [n_fm, -1])

    dx = tf.matmul(tf.transpose(weights2d, perm=[1, 0]), dO)
    dx = col_to_img(dx, inp.shape, h_w, w_w, fm_h, fm_w, pad, stride)

    dw = tf.matmul(dO, tf.transpose(inp2d, perm=[1, 0]))
    dw = tf.reshape(dw, [tf.shape(weights)])
    return dx, dw

def pool_backprop(delta_out, inp, p_h, p_w, stride, pad=0):
    N, D, h, w = tf.shape(inp)
    o_h = tf.constant(((h + 2 * pad - p_h) / stride + 1), dtype=tf.int32)
    o_w = tf.constant(((w + 2 * pad - p_w) / stride + 1), dtype=tf.int32)

    inp_reshaped = tf.reshape(inp, [-1, 1, h, w])
    #inp_col = img_to_col(inp_reshaped, p_h, p_w, o_h, o_w, pad, stride)
    inp_col = tf.py_func(im2col, [inp_reshaped, p_h, p_w, o_h, o_w, pad, stride], tf.float32)

    err = tf.reshape(tf.transpose(delta_out, perm=[2, 3, 0, 1]), -1)

    max_indices = tf.argmax(inp_col, axis=0)
    back = tf.zeros_like(inp_col)
    back[max_indices, tf.range(tf.shape(inp_col)[1])] = err                     #***************************************************************
    back = col_to_img(back, tf.shape(inp_reshaped), p_h, p_w, o_h, o_w, pad, stride)
    dx = tf.reshape(back, tf.shape(inp))
    return dx

def batch_norm_backprop(delta_out, cache):
    N, D, H, W = tf.shape(delta_out)
    xmu, inv_std, x_hat, gamma = cache

    M = N*H*W
    dO = tf.reshape(tf.transpose(delta_out, [0, 2, 3, 1]), [M, D])

    db = tf.reduce_sum(dO, axis=0)
    dg = tf.reduce_sum(x_hat * dO, axis=0)

    dx_hat = dO * gamma

    dx1 = x_hat * tf.reduce_sum(dx_hat * x_hat, axis=0)
    dx2 = M * dx_hat - tf.reduce_sum(dx_hat, axis=0)
    dx = 1. / M * inv_std * (dx2 - dx1)

    dx = tf.transpose(tf.reshape(dx, [N, H, W, D]), [0, 3, 1, 2])

    return dx, db, dg

def full_conn_backprop(delta_out, inp, weights):
    dx = tf.matmul(delta_out, tf.transpose(weights, perm=[1, 0]))
    dw = tf.matmul(tf.transpose(inp, perm=[1, 0]), delta_out)
    return dx, dw

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
    _, i_c, i_h, i_w = inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]

    # z axis (channel)
    z = tf.reshape(tf.tile(tf.reshape(tf.range(i_c), [-1, 1]), [1, f_h * f_w]), [-1, 1])

    # y index
    y0 = tf.tile(tf.reshape(tf.tile(tf.reshape(tf.range(f_h), [-1, 1]), [1, f_w]), [-1]), [i_c])
    y1 = stride * tf.reshape(tf.tile(tf.reshape(tf.range(o_h), [-1, 1]), [1, o_w]), [-1])
    y = tf.reshape(y0, [-1, 1]) + tf.reshape(y1, [1, -1])

    # x index
    x0 = tf.tile(tf.range(f_w), [f_h * i_c])
    x1 = stride * tf.tile(tf.range(o_w), [o_h])
    x = tf.reshape(x0, [-1, 1]) + tf.reshape(x1, [1, -1])

    return z, y, x


def img_to_col(inp, f_h, f_w, o_h, o_w, pad, stride):
    z, y, x = get_col_indices(tf.shape(inp), f_h, f_w, o_h, o_w, stride)

    inp_padded = tf.pad(inp, [[0, 0], [0, 0], [pad, pad], [pad, pad]], mode='CONSTANT')

    indices = tf.range(tf.shape(inp)[0])
    col = tf.gather_nd(inp_padded, z, y, x)
    col = tf.transpose(col, perm=[1, 2, 0])
    return tf.reshape(col, [(f_h * f_w * tf.shape(inp)[1]), -1])


def col_to_img(col, inp_shape, f_h, f_w, o_h, o_w, pad, stride):
    N, i_c, i_h, i_w = inp_shape
    z, y, x = get_col_indices(inp_shape,f_h, f_w, o_h, o_w, stride)

    h_p, w_p = i_h + 2 * pad, i_w + 2 * pad
    img = tf.zeros((N, i_c, h_p, w_p), dtype=float)
    cols_reshaped = tf.reshape(col, [i_c*f_h*f_w, -1, N])
    cols_reshaped = tf.transpose(cols_reshaped, [2, 0, 1])

    def add_at(i, z, x, y, c):
        np.add.at(i, (slice(None), z, y, x), c)
        img = np.copy(i)
        return img.astype(np.float32)

    img = tf.py_func(add_at, [img, z, y, x, cols_reshaped], tf.float32)

    return img[:, :, pad:-pad, pad:-pad]


def intersection_over_union(box1, box2, box1_cell, box2_cell, img_width, img_height):
    I = _find_intersection(box1, box2, box1_cell, box2_cell, img_width, img_height)
    U = (box1[3] * box1[4] + box2[3] * box2[4]) - I
    return I / U


def _find_intersection(box1, box2, box1_cell, box2_cell, width, height):
    TL1, TR1, BL1, BR1 = _find_corners(box1, box1_cell)
    b1 = tf.zeros((height, width))
    b1x1 = TL1[1]
    b1x2 = TR1[1]
    b1y1 = TL1[0]
    b1y2 = BL1[0]
    b1[b1y1:b1y2, b1x1:b1x2] = 1

    TL2, TR2, BL2, BR2 = _find_corners(box2, box2_cell)
    b2 = tf.zeros((height, width))
    b2x1 = TL2[1]
    b2x2 = TR2[1]
    b2y1 = TL2[0]
    b2y2 = BL2[0]
    b2[b2y1:b2y2, b2x1:b2x2] = 1

    return tf.reduce_sum(b1 * b2)


def _find_corners(box, cell):
    bb_h = tf.floor(box[3] * IMG_HEIGHT)
    bb_w = tf.floor(box[4] * IMG_WIDTH)
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

def find_midpoint(box):
    img_xm = box[0] + (box[2] / 2)
    img_ym = box[1] + (box[3] / 2)
    c_x = (img_xm % CELL_WIDTH) / CELL_WIDTH
    c_y = (img_ym % CELL_HEIGHT) / CELL_HEIGHT
    cell_x = floor(box[0] / CELL_WIDTH)
    cell_y = floor(box[1] / CELL_HEIGHT)
    cell_num = cell_y * GRID_WIDTH + cell_x
    bb_width = box[2] / IMG_WIDTH
    bb_height = box[3] / IMG_HEIGHT
    return cell_num, c_x, c_y, bb_width, bb_height


def rescale_labels(x, y, box_w, box_h, img_w, img_h):
    s_w = floor(IMG_WIDTH / img_w)
    s_h = floor(IMG_HEIGHT / img_h)
    Nx = s_w * x
    Ny = s_h * y
    Nw = s_w * box_w
    Nh = s_h * box_h
    return Nx, Ny, Nw, Nh


def non_max_suppression(bounding_boxes, conf_threshold, IoU_threshold):
    bounding_boxes = tf.reshape(bounding_boxes, [-1, 5])
    bnd_boxes = bounding_boxes
    predictions = tf.zeros(0, dtype=tf.int32)
    i = 0
    while i < tf.shape(bnd_boxes)[0]:
        if bnd_boxes[i][0] < conf_threshold:
            bnd_boxes[i][0] = 0
            i = i + 1
        else:
            p = tf.argmax(tf.transpose(bnd_boxes, perm=[1, 0])[0], axis=0)
            predictions = tf.reshape(tf.concat(predictions, p), [-1])
            bnd_boxes[predictions[-1]][0] = 0
            for b in range(bnd_boxes):
                if bnd_boxes[b][0] != 0:
                    overlap = intersection_over_union(bnd_boxes[b], bnd_boxes[predictions[-1]], b, predictions[-1], IMG_HEIGHT, IMG_WIDTH)
                    if overlap >= IoU_threshold:
                        bnd_boxes[b][0] = 0
                        i = i + 1
    return tf.gather(bounding_boxes, predictions)


def initialize_weights(params):
    FM, C, H, W, _, _ = params
    w = tf.truncated_normal([FM, C, H, W], stddev=0.01)
    weights = tf.Variable(w, name='weights')
    bn_shape = [1, FM, 1, 1]
    beta = tf.Variable(tf.zeros(bn_shape), name='beta')
    #gamma = tf.Variable(tf.ones(bn_shape), name='gamma')
    #mean = tf.Variable(tf.zeros(bn_shape), trainable=False, name="bn_mean")
    #var = tf.Variable(tf.ones(bn_shape), trainable=False, name='bn_var')
    return weights, beta, None, None, None #gamma, mean, var

def update_weights(weights, gradients, learning_rate, batch_size):
    return weights - (learning_rate / batch_size * gradients)


def normalize(inp, epsilon=1e-8):
    mean, var = tf.nn.moments(inp, axes=[1, 2], keep_dims=True)
    return (inp - mean) / (tf.sqrt(var + epsilon))

def add_layer_summaries(layer, conv, relu, batch_norm, pool=None):
    tf.summary.histogram('conv_' + layer, conv)
    tf.summary.histogram('relu_' + layer , relu)
    tf.summary.histogram('batch_norm_' + layer, batch_norm)

    if pool is not None:
        tf.summary.histogram('pool', pool)

def add_weight_summaries(name, weights):
    with tf.name_scope(name):
        mean = tf.reduce_mean(weights)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(weights - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(weights))
        tf.summary.scalar('min', tf.reduce_min(weights))