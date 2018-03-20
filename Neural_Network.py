import tensorflow as tf
import numpy as np
from math import sqrt
import Operations_tf as op

def create_facial_rec(inp, architecture, training=False):
    '''
    Constructs the Neural Net Graph
    :param inp: Input Tensor
    :param architecture: Architecture of the Network
    :param training: Toggle Training
    :return: Last node on graph
    '''

    '''   LAYER 1   '''
    FM, C, H, W, _, pad = architecture['conv1']
    p_h, p_w, stride, _ = architecture['pool1']

    with tf.name_scope('layer_1'):
        w_conv1, beta1, gamma1, mean1, var1 = op.initialize_weights(architecture['conv1'])
        w = tf.transpose(w_conv1, perm=[2, 3, 1, 0])
        conv1 = tf.nn.conv2d(inp, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
        relu1 = op.relu(conv1)
        batch_norm1 = op.batch_normalize(relu1, beta1, gamma1,
                                         mean1, var1, training=training)
        pool1 = op.pool(batch_norm1[0], p_h, p_w, stride)

        mean1 = batch_norm1[2]
        var1 = batch_norm1[3]

        #op.add_weight_summaries('mean', mean1)
        #op.add_weight_summaries('var', var1)

    # op.add_layer_summaries('1', conv1, relu1, batch_norm1[0], pool=pool1)

    '''   LAYER 2   '''
    FM, C, H, W, _, pad = architecture['conv2']
    p_h, p_w, stride, _ = architecture['pool2']
    with tf.name_scope('layer_2'):
        w_conv2, beta2, gamma2, mean2, var2 = op.initialize_weights(architecture['conv2'])
        w = tf.transpose(w_conv2, perm=[2, 3, 1, 0])
        conv2 = tf.nn.conv2d(pool1, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
        relu2 = op.relu(conv2)
        batch_norm2 = op.batch_normalize(relu2, beta2, gamma2,
                                         mean2, var2, training=training)
        pool2 = op.pool(batch_norm2[0], p_h, p_w, stride)

        mean2 = batch_norm2[2]
        var2 = batch_norm2[3]

        #op.add_weight_summaries('mean', mean2)
        #op.add_weight_summaries('var', var2)

    # op.add_layer_summaries('2', conv2, relu2, batch_norm2[0], pool=pool2)

    '''   LAYER 3   '''
    FM, C, H, W, _, pad = architecture['conv3']
    p_h, p_w, stride, _ = architecture['pool3']
    with tf.name_scope('layer_3'):
        w_conv3, beta3, gamma3, mean3, var3 = op.initialize_weights(architecture['conv3'])
        w = tf.transpose(w_conv3, perm=[2, 3, 1, 0])
        conv3 = tf.nn.conv2d(pool2, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
        relu3 = op.relu(conv3)
        batch_norm3 = op.batch_normalize(relu3, beta3, gamma3,
                                         mean3, var3, training=training)
        pool3 = op.pool(batch_norm3[0], p_h, p_w, stride)

        mean3 = batch_norm3[2]
        var3 = batch_norm3[3]

        #op.add_weight_summaries('mean', mean3)
        #op.add_weight_summaries('var', var3)

    # op.add_layer_summaries('3', conv3, relu3, batch_norm3[0])

    '''   LAYER 4   '''
    FM, C, H, W, _, pad = architecture['conv4']
    p_h, p_w, stride, _ = architecture['pool4']
    with tf.name_scope('layer_4'):
        w_conv4, beta4, gamma4, mean4, var4 = op.initialize_weights(architecture['conv4'])
        w = tf.transpose(w_conv4, perm=[2, 3, 1, 0])
        conv4 = tf.nn.conv2d(pool3, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
        relu4 = op.relu(conv4)
        batch_norm4 = op.batch_normalize(relu4, beta4, gamma4,
                                         mean4, var4, training=training)
        pool4 = op.pool(batch_norm4[0], p_h, p_w, stride)

        mean4 = batch_norm4[2]
        var4 = batch_norm4[3]

        #op.add_weight_summaries('mean', mean4)
        #op.add_weight_summaries('var', var4)

    # op.add_layer_summaries('4', conv4, relu4, batch_norm4[0])

    '''   LAYER 5   '''
    FM, C, H, W, _, pad = architecture['conv5']
    with tf.name_scope('layer_5'):
        w_conv5, beta5, gamma5, mean5, var5 = op.initialize_weights(architecture['conv5'])
        w = tf.transpose(w_conv5, perm=[2, 3, 1, 0])
        conv5 = tf.nn.conv2d(pool4, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
        relu5 = op.relu(conv5)
        batch_norm5 = op.batch_normalize(relu5, beta5, gamma5,
                                         mean5, var5, training=training)
        mean5 = batch_norm5[2]
        var5 = batch_norm5[3]

        #op.add_weight_summaries('mean', mean5)
        #op.add_weight_summaries('var', var5)

    # op.add_layer_summaries('5', conv5, relu5, batch_norm5[0], pool=pool5)

    '''   FULLY CONNECTED LAYER   '''
    fc_in, fc_out = architecture['full']
    with tf.name_scope('full_conn_layer'):
        w_full = tf.Variable(tf.truncated_normal([fc_in, fc_out], stddev=0.001), name='weights')
        b_full = tf.Variable(tf.zeros([fc_out]), name='biases')
        N, D = tf.shape(batch_norm5[0])[0], tf.shape(batch_norm5[0])[1]
        H, W = tf.shape(batch_norm5[0])[2], tf.shape(batch_norm5[0])[3]
        flatten = tf.reshape(batch_norm5[0], [-1, D * H * W])
        full_conn = op.full_conn(flatten, w_full, b_full)
    tf.summary.histogram('full_conn', full_conn)

    #prediction = full_conn                   # for training
    prediction = tf.nn.softmax(full_conn)
    return prediction


def create_emotion_rec(inp, architecture, training=False):
    pass

def update_weights(weights, gradients, learning_rate, batch_size):
    for w in weights:
        weights[w] = op.update_weights(weights[w], gradients[w], learning_rate, batch_size)
    return weights

def load_model(sess, model_filename):
    saver = tf.train.Saver()
    saver.restore(sess, model_filename)
