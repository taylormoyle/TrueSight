import tensorflow as tf
import numpy as np
from math import sqrt
import Operations_tf as op

class Neural_Network:

    def __init__(self, model_type, infos, hypers):
        #model_type can specify which application we want to customize our operations for.
        self.init_facial_rec_weights(infos)
        if model_type == "facial_recognition":
            #_, self.conv_thresh, self.IoU_thresh = hypers
            #self._init_facial_rec_weights(infos)
            #self.create_facial_rec(infos)
            pass

        if model_type == "emotion_recognition":
            #self.create_emotion_rec(infos)
            pass

    # Create the shell of the facial recognition neural network.
    def create_facial_rec(self, infos, inp, weights):
        pass


    def create_emotion_rec(self, infos, inp):
        pass


    def forward_prop(self, infos, inp, training=False):
        n_fm, f_h, f_w, _, pad1 = infos[0]
        n_fm, f_h, f_w, _, pad2 = infos[1]
        n_fm, f_h, f_w, _, pad3 = infos[3]
        n_fm, f_h, f_w, _, pad4 = infos[4]
        n_fm, f_h, f_w, _, pad5 = infos[6]
        n_fm, f_h, f_w, _, pad6 = infos[8]
        n_fm, f_h, f_w, _, pad7 = infos[10]
        n_fm, f_h, f_w, _, pad8 = infos[12]
        n_fm, f_h, f_w, _, pad9 = infos[13]
        n_fm, f_h, f_w, _, pad10 = infos[14]

        _, p_h1, p_w1, stride1, _ = infos[2]
        _, p_h2, p_w2, stride2, _ = infos[5]
        _, p_h3, p_w3, stride3, _ = infos[7]
        _, p_h4, p_w4, stride4, _ = infos[9]
        _, p_h5, p_w5, stride5, _ = infos[11]

        # normalize images
        inp = op.normalize(inp)

        '''   LAYER 1   '''
        with tf.name_scope('layer_1'):
            #conv1 = op.convolve(inp, self.w_conv1, pad=pad1)
            w = tf.transpose(self.w_conv1, perm=[2, 3, 1, 0])
            conv1 = tf.nn.conv2d(inp, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
            relu1 = op.relu(conv1)
            batch_norm1 = op.batch_normalize(relu1, self.w_bnb1, self.w_bng1, self.running_mean_var_bn1, training=training)

        op.add_layer_summaries('1', conv1, relu1, batch_norm1[0])

        '''   LAYER 2   '''
        with tf.name_scope('layer_2'):
            #conv2 = op.convolve(batch_norm1[0], self.w_conv2, pad=pad2)
            w = tf.transpose(self.w_conv2, perm=[2, 3, 1, 0])
            conv2 = tf.nn.conv2d(batch_norm1[0], w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
            relu2 = op.relu(conv2)
            batch_norm2 = op.batch_normalize(relu2, self.w_bnb2, self.w_bng2, self.running_mean_var_bn2, training=training)
            pool2 = op.pool(batch_norm2[0], p_h1, p_w1, stride1)

        op.add_layer_summaries('2', conv2, relu2, batch_norm2[0], pool=pool2)

        '''   LAYER 3   '''
        with tf.name_scope('layer_3'):
            #conv3 = op.convolve(pool2, self.w_conv3, pad=pad3)
            w = tf.transpose(self.w_conv3, perm=[2, 3, 1, 0])
            conv3 = tf.nn.conv2d(pool2, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
            relu3 = op.relu(conv3)
            batch_norm3 = op.batch_normalize(relu3, self.w_bnb3, self.w_bng3, self.running_mean_var_bn3, training=training)

        op.add_layer_summaries('3', conv3, relu3, batch_norm3[0])

        '''   LAYER 4   '''
        with tf.name_scope('layer_4'):
            #conv4 = op.convolve(batch_norm3[0], self.w_conv4, pad=pad4)
            w = tf.transpose(self.w_conv4, perm=[2, 3, 1, 0])
            conv4 = tf.nn.conv2d(batch_norm3[0], w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
            relu4 = op.relu(conv4)
            batch_norm4 = op.batch_normalize(relu4, self.w_bnb4, self.w_bng4, self.running_mean_var_bn4, training=training)
            pool4 = op.pool(batch_norm4[0], p_h2, p_w2, stride2)

        op.add_layer_summaries('4', conv4, relu4, batch_norm4[0], pool=pool4)

        '''   LAYER 5   '''
        with tf.name_scope('layer_5'):
            #conv5 = op.convolve(pool4, self.w_conv5, pad=pad5)
            w = tf.transpose(self.w_conv5, perm=[2, 3, 1, 0])
            conv5 = tf.nn.conv2d(pool4, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
            relu5 = op.relu(conv5)
            batch_norm5 = op.batch_normalize(relu5, self.w_bnb5, self.w_bng5, self.running_mean_var_bn5, training=training)
            pool5 = op.pool(batch_norm5[0], p_h3, p_w3, stride3)

        op.add_layer_summaries('5', conv5, relu5, batch_norm5[0], pool=pool5)

        '''   LAYER 6   '''
        with tf.name_scope('layer_6'):
            #conv6 = op.convolve(pool5, self.w_conv6, pad=pad6)
            w = tf.transpose(self.w_conv6, perm=[2, 3, 1, 0])
            conv6 = tf.nn.conv2d(pool5, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
            relu6 = op.relu(conv6)
            batch_norm6 = op.batch_normalize(relu6, self.w_bnb6, self.w_bng6, self.running_mean_var_bn6, training=training)
            pool6 = op.pool(batch_norm6[0], p_h4, p_w4, stride4)

        op.add_layer_summaries('6', conv6, relu6, batch_norm6[0], pool=pool6)

        """
        '''   LAYER 7   '''
        #conv7 = op.convolve(pool6, self.w_conv7, pad=pad7)
        w = tf.transpose(self.w_conv7, perm=[2, 3, 1, 0])
        conv7 = tf.nn.conv2d(pool6, w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
        relu7 = op.relu(conv7)
        batch_norm7 = op.batch_normalize(relu7, self.w_bnb7, self.w_bng7, self.running_mean_var_bn7, training=training)
        pool7 = op.pool(batch_norm7[0], p_h5, p_w5, stride5)

        '''   LAYER 8   '''
        conv8 = op.convolve(pool7, self.w_conv8, pad=pad8)
        relu8 = op.relu(conv8)
        batch_norm8 = op.batch_normalize(relu8, self.w_bnb8, self.w_bng8, self.running_mean_var_bn8, training=training)

        '''   LAYER 9   '''
        conv9 = op.convolve(batch_norm8[0], self.w_conv9, pad=pad9)
        relu9 = op.relu(conv9)
        batch_norm9 = op.batch_normalize(relu9, self.w_bnb9, self.w_bng9, self.running_mean_var_bn9, training=training)

        '''   LAYER 10   '''
        conv10 = op.convolve(batch_norm9[0], self.w_conv10, pad=pad10)
        relu10 = op.relu(conv10)
        #prediction = op.non_max_suppression(relu10, self.conv_thresh, self.IoU_thresh)
        """

        '''   FULLY CONNECTED LAYER   '''
        with tf.name_scope('full_conn_layer'):
            N, D, H, W = tf.shape(pool6)[0], tf.shape(pool6)[1], tf.shape(pool6)[2], tf.shape(pool6)[3]
            flatten = tf.reshape(pool6, [-1, D * H * W])
            full_conn = op.full_conn(flatten, self.w_full)
            #prediction = op.sigmoid(full_conn)
        tf.summary.histogram('full_conn', full_conn)
        prediction = full_conn

        """ 
        if training:
            cache = {  # 'c10': conv10, 'c9': conv9, 'c8': conv8, 'c7': conv7, 'c6': conv6, 'c5': conv5, 'c4': conv4,
                # 'c3': conv3, 'c2': conv2, 'c1': conv1, 'r10': relu10, 'r9': relu9, 'r8': relu8, 'r7': relu7,
                # 'r6': relu6, 'r5': relu5, 'r4': relu4, 'r3': relu3, 'r2': relu2, 'r1': relu1, 'p7': pool7,
                # 'p6': pool6, 'p5': pool5,
                'c2': conv2, 'c1': conv1, 'c3': conv3, 'r3': relu3, 'r2': relu2, 'r1': relu1, 'bn1': batch_norm1,
                'bn2': batch_norm2, 'bn3': batch_norm3, 'p2': pool2, 'inp': inp, 'fc': full_conn, 'fl': flatten
                # 'c1': conv1, 'bn1': batch_norm1, 'inp': inp, 'fc': full_conn, 'fl': flatten
            }

            for b in self.running_mean_var:
                self.running_mean_var[b] = cache[b][2]

            return prediction, cache
        """
        return prediction

    def backward_prop(self, cache, output, labels, weights, infos):
        n_fm, f_h, f_w, _, pad1 = infos[0]
        n_fm, f_h, f_w, _, pad2 = infos[1]
        n_fm, f_h, f_w, _, pad3 = infos[3]
        n_fm, f_h, f_w, _, pad4 = infos[4]
        n_fm, f_h, f_w, _, pad5 = infos[6]
        n_fm, f_h, f_w, _, pad6 = infos[8]
        n_fm, f_h, f_w, _, pad7 = infos[10]
        n_fm, f_h, f_w, _, pad8 = infos[12]
        n_fm, f_h, f_w, _, pad9 = infos[13]
        n_fm, f_h, f_w, _, pad10 = infos[14]

        _, p_h1, p_w1, stride1, _ = infos[2]
        _, p_h2, p_w2, stride2, _ = infos[5]
        _, p_h3, p_w3, stride3, _ = infos[7]
        _, p_h4, p_w4, stride4, _ = infos[9]
        _, p_h5, p_w5, stride5, _ = infos[11]

        w = weights

        batch_size, _ = output.shape
        error = op.mse_prime(output, labels)
        delta_out = error * op.sigmoid_prime(cache['fc']) / batch_size

        '''   FULLY CONNECTED LAYER   '''
        full_dx, full_dw = op.full_conn_backprop(delta_out, cache['fl'], w['full'])
        full_dx = full_dx.reshape(cache['bn3'][0].shape)
        """
        '''   LAYER 10   '''
        relu_error = op.relu_prime(cache['c10']) * full_dx #* error
        conv_dx, conv10_dw = op.convolve_backprop(relu_error, cache['r9'], w['conv10'], pad=pad10)

        '''   LAYER 9   '''
        #batch_norm_dx, batch_norm9_dbg = op.batch_norm_backprop(conv_dx, cache[1][1])
        relu_error = op.relu_prime(cache['c9']) * conv_dx#* batch_norm_dx
        conv_dx, conv9_dw = op.convolve_backprop(relu_error, cache['r8'], w['conv9'], pad=pad9)

        '''   LAYER 8   '''
        #batch_norm_dx, batch_norm8_dbg = op.batch_norm_backprop(conv_dx, cache[3][1])
        relu_error = op.relu_prime(cache['c8']) * conv_dx #* batch_norm_dx
        conv_dx, conv8_dw = op.convolve_backprop(relu_error, cache['p7'], w['conv8'], pad=pad8)

        '''   LAYER 7   '''
        pool = op.pool_backprop(conv_dx, cache['r7'], p_h5, p_w5, stride5)
        #batch_norm_dx, batch_norm7_dbg = op.batch_norm_backprop(pool, cache[3][1])
        relu_error = op.relu_prime(cache['c7']) * pool #* batch_norm_dx
        conv_dx, conv7_dw = op.convolve_backprop(relu_error, cache['p6'], w['conv7'], pad=pad7)

        '''   LAYER 6   '''
        pool = op.pool_backprop(conv_dx, cache['r6'], p_h4, p_w4, stride4)
        #batch_norm_dx, batch_norm6_dbg = op.batch_norm_backprop(pool, cache[6][1])
        relu_error = op.relu_prime(cache['c6']) * pool #* batch_norm_dx
        conv_dx, conv6_dw = op.convolve_backprop(relu_error, cache['p5'], w['conv6'], pad=pad6)

        '''   LAYER 5   '''
        pool = op.pool_backprop(conv_dx, cache['r5'], p_h3, p_w3, stride3)
        #batch_norm_dx, batch_norm5_dbg = op.batch_norm_backprop(pool, cache[9][1])
        relu_error = op.relu_prime(cache['c5']) * pool #* batch_norm_dx
        conv_dx, conv5_dw = op.convolve_backprop(relu_error, cache['p4'], w['conv5'], pad=pad5)

        '''   LAYER 4   '''
        pool = op.pool_backprop(full_dx, cache['r4'], p_h2, p_w2, stride2)
        #batch_norm_dx, batch_norm4_dbg = op.batch_norm_backprop(pool, cache[12][1])
        relu_error = op.relu_prime(cache['c4']) * pool #* batch_norm_dx
        conv_dx, conv4_dw = op.convolve_backprop(relu_error, cache['r3'], w['conv4'], pad=pad4)
        """
        '''   LAYER 3   '''
        batch_norm3_dx, batch_norm3_db, batch_norm3_dg = op.batch_norm_backprop(full_dx, cache['bn3'][1])
        relu_error = op.sigmoid_prime(cache['c3']) * batch_norm3_dx
        conv_dx, conv3_dw = op.convolve_backprop(relu_error, cache['p2'], w['conv3'], pad=pad3)

        '''   LAYER 2   '''
        pool = op.pool_backprop(conv_dx, cache['bn2'][0], p_h1, p_w1, stride1)
        batch_norm2_dx, batch_norm2_db, batch_norm2_dg = op.batch_norm_backprop(pool, cache['bn2'][1])
        relu_error = op.sigmoid_prime(cache['c2']) * batch_norm2_dx
        conv_dx, conv2_dw = op.convolve_backprop(relu_error, cache['bn1'][0], w['conv2'], pad=pad2)

        '''   LAYER 1   '''
        batch_norm1_dx, batch_norm1_db, batch_norm1_dg = op.batch_norm_backprop(conv_dx, cache['bn1'][1])
        relu_error = op.sigmoid_prime(cache['c1']) * batch_norm1_dx
        _, conv1_dw = op.convolve_backprop(relu_error, cache['inp'], w['conv1'], pad=pad1)

        gradients = {'conv1': conv1_dw, 'conv2': conv2_dw, 'conv3': conv3_dw,  # 'conv4': conv4_dw, 'conv5': conv5_dw,
                     # 'conv6': conv6_dw, 'conv7': conv7_dw, 'conv8': conv8_dw, 'conv9': conv9_dw, 'conv10': conv10_dw,
                     'bnb1': batch_norm1_db, 'bnb2': batch_norm2_db, 'bnb3': batch_norm3_db, 'bng1': batch_norm1_dg,
                     'bng2': batch_norm2_dg, 'bng3': batch_norm3_dg,
                     'full': full_dw
                     }

        return gradients

    def init_facial_rec_weights(self, infos):
        with tf.name_scope('layer_1_weights'):
            n_fm1, f_h, f_w, _, _ = infos[0]
            self.w_conv1 = op.initialize_conv_weights((n_fm1, 3, f_h, f_w), 'w_conv1')
            self.w_bnb1 = tf.Variable(tf.zeros([n_fm1]), name='w_bnb1')
            self.w_bng1 = tf.Variable(tf.ones([n_fm1]), name='w_bng1')
            self.running_mean_var_bn1 = tf.Variable([tf.zeros(n_fm1), tf.ones(n_fm1)],
                                                    trainable=False, name="running_bn1")

        op.add_weight_summaries('conv_1_weights', self.w_conv1)
        op.add_weight_summaries('beta_1', self.w_bnb1)
        op.add_weight_summaries('gamma_1', self.w_bng1)
        op.add_weight_summaries('running_mean_var_1', self.running_mean_var_bn1)

        with tf.name_scope('layer_2_weights'):
            n_fm2, f_h, f_w, _, _ = infos[1]
            self.w_conv2 = op.initialize_conv_weights((n_fm2, n_fm1, f_h, f_w), 'w_conv2')
            self.w_bnb2 = tf.Variable(tf.zeros([n_fm2]), name='w_bnb2')
            self.w_bng2 = tf.Variable(tf.ones([n_fm2]), name='w_bng2')
            self.running_mean_var_bn2 = tf.Variable([tf.zeros(n_fm2), tf.ones(n_fm2)],
                                                    trainable=False, name="running_bn2")

        op.add_weight_summaries('conv_2_weights', self.w_conv2)
        op.add_weight_summaries('beta_2', self.w_bnb2)
        op.add_weight_summaries('gamma_2', self.w_bng2)
        op.add_weight_summaries('running_mean_var_2', self.running_mean_var_bn2)

        with tf.name_scope('layer_3_weights'):
            n_fm3, f_h, f_w, _, _ = infos[3]
            self.w_conv3 = op.initialize_conv_weights((n_fm3, n_fm2, f_h, f_w), 'w_conv3')
            self.w_bnb3 = tf.Variable(tf.zeros([n_fm3]), name='w_bnb3')
            self.w_bng3 = tf.Variable(tf.ones([n_fm3]), name='w_bng3')
            self.running_mean_var_bn3 = tf.Variable([tf.zeros(n_fm3), tf.ones(n_fm3)],
                                                    trainable=False, name="running_bn3")

        op.add_weight_summaries('conv_3_weights', self.w_conv3)
        op.add_weight_summaries('beta_3', self.w_bnb3)
        op.add_weight_summaries('gamma_3', self.w_bng3)
        op.add_weight_summaries('running_mean_var_3', self.running_mean_var_bn3)

        with tf.name_scope('layer_4_weights'):
            n_fm4, f_h, f_w, _, _ = infos[4]
            self.w_conv4 = op.initialize_conv_weights((n_fm4, n_fm3, f_h, f_w), 'w_conv4')
            self.w_bnb4 = tf.Variable(tf.zeros([n_fm4]), name='w_bnb4')
            self.w_bng4 = tf.Variable(tf.ones([n_fm4]), name='w_bng4')
            self.running_mean_var_bn4 = tf.Variable([tf.zeros(n_fm4), tf.ones(n_fm4)],
                                                    trainable=False, name="running_bn4")

        op.add_weight_summaries('conv_4_weights', self.w_conv4)
        op.add_weight_summaries('beta_4', self.w_bnb4)
        op.add_weight_summaries('gamma_4', self.w_bng4)
        op.add_weight_summaries('running_mean_var_4', self.running_mean_var_bn4)

        with tf.name_scope('layer_5_weights'):
            n_fm5, f_h, f_w, _, _ = infos[6]
            self.w_conv5 = op.initialize_conv_weights((n_fm5, n_fm4, f_h, f_w), 'w_conv5')
            self.w_bnb5 = tf.Variable(tf.zeros([n_fm5]), name='w_bnb5')
            self.w_bng5 = tf.Variable(tf.ones([n_fm5]), name='w_bng5')
            self.running_mean_var_bn5 = tf.Variable([tf.zeros(n_fm5), tf.ones(n_fm5)],
                                                    trainable=False, name="running_bn5")

        op.add_weight_summaries('conv_5_weights', self.w_conv5)
        op.add_weight_summaries('beta_5', self.w_bnb5)
        op.add_weight_summaries('gamma_5', self.w_bng5)
        op.add_weight_summaries('running_mean_var_5', self.running_mean_var_bn5)

        with tf.name_scope('layer_6_weights'):
            n_fm6, f_h, f_w, _, _ = infos[8]
            self.w_conv6 = op.initialize_conv_weights((n_fm6, n_fm5, f_h, f_w), 'w_conv6')
            self.w_bnb6 = tf.Variable(tf.zeros([n_fm6]), name='w_bnb6')
            self.w_bng6 = tf.Variable(tf.ones([n_fm6]), name='w_bng6')
            self.running_mean_var_bn6 = tf.Variable([tf.zeros(n_fm6), tf.ones(n_fm6)],
                                                    trainable=False, name="running_bn6")

        op.add_weight_summaries('conv_6_weights', self.w_conv6)
        op.add_weight_summaries('beta_6', self.w_bnb6)
        op.add_weight_summaries('gamma_6', self.w_bng6)
        op.add_weight_summaries('running_mean_var_6', self.running_mean_var_bn6)

        """
        n_fm7, f_h, f_w, _, _ = infos[10]
        self.w_conv7 = op.initialize_conv_weights((n_fm7, n_fm6, f_h, f_w), 'w_conv7')
        self.w_bnb7 = op.initialize_2d_weights((n_fm7,1), 'w_bnb7')
        self.w_bng7 = op.initialize_2d_weights((n_fm7,1), 'w_bng7')
        self.running_mean_var_bn7 = tf.Variable([[0.],[1.]], trainable=False, validate_shape=False)

        n_fm8, f_h, f_w, _, _ = infos[12]
        self.w_conv8 = op.initialize_conv_weights((n_fm8, n_fm7, f_h, f_w), 'w_conv8')
        self.w_bnb8 = op.initialize_2d_weights((n_fm8,1), 'w_bnb8')
        self.w_bng8 = op.initialize_2d_weights((n_fm8,1), 'w_bng8')
        self.running_mean_var_bn8 = tf.Variable([[0.],[1.]], trainable=False, validate_shape=False)

        n_fm9, f_h, f_w, _, _ = infos[13]
        self.w_conv9 = op.initialize_conv_weights((n_fm9, n_fm8, f_h, f_w), 'w_conv9')
        self.w_bnb9 = op.initialize_2d_weights((n_fm9,1), 'w_bnb9')
        self.w_bng9 = op.initialize_2d_weights((n_fm9,1), 'w_bng9')
        self.running_mean_var_bn9 = tf.Variable([[0.],[1.]], trainable=False, validate_shape=False)

        n_fm10, f_h, f_w, _, _ = infos[14]
        self.w_conv10 = op.initialize_conv_weights((n_fm10, n_fm9, f_h, f_w), 'w_conv10')
        """
        with tf.name_scope('full_con_weights'):
            fc_in = 13 * 13 * 64
            self.w_full = op.initialize_2d_weights((fc_in, 20), 'w_full')
        op.add_weight_summaries('full_conn', self.w_full)

    def update_weights(self, weights, gradients, learning_rate, batch_size):
        for w in weights:
            weights[w] = op.update_weights(weights[w], gradients[w], learning_rate, batch_size)
        return weights
