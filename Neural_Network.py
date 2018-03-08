import tensorflow as tf
import numpy as np
from math import sqrt
import Operations_tf as op

class Neural_Network:

    def __init__(self, model_type, infos, hypers, training=False):
        #model_type can specify which application we want to customize our operations for.

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

        '''   LAYER 1   '''
        conv1 = op.convolve(inp, w['conv1'], pad=pad1)
        batch_norm1 = op.batch_normalize(conv1, w['bn1'])
        relu1 = op.relu(batch_norm1)

        '''   LAYER 2   '''
        conv2 = op.convolve(relu1, w['conv2'], pad=pad2)
        batch_norm2 = op.batch_normalize(conv2, w['bn2'])
        relu2 = op.relu(batch_norm2)
        pool2 = op.pool(relu2, p_h1, p_w1, stride1)

        '''   LAYER 3   '''
        conv3 = op.convolve(pool2, w['conv3'], pad=pad3)
        batch_norm3 = op.batch_normalize(conv3, w['bn3'])
        relu3 = op.relu(batch_norm3)

        '''   LAYER 4   '''
        conv4 = op.convolve(relu3, w['conv4'], pad=pad4)
        batch_norm4 = op.batch_normalize(conv4, w['bn4'])
        relu4 = op.relu(batch_norm4)
        pool4 = op.pool(relu4, p_h2, p_w2, stride2)

        '''   LAYER 5   '''
        conv5 = op.convolve(pool4, w['conv5'], pad=pad5)
        batch_norm5 = op.batch_normalize(conv5, w['bn5'])
        relu5 = op.relu(batch_norm5)
        pool5 = op.pool(relu5, p_h3, p_w3, stride3)

        '''   LAYER 6   '''

        conv6 = op.convolve(pool5, w['conv6'], pad=pad6)
        batch_norm6 = op.batch_normalize(conv6, w['bn6'])
        relu6 = op.relu(batch_norm6)
        pool6 = op.pool(relu6, p_h4, p_w4, stride4)

        '''   LAYER 7   '''
        conv7 = op.convolve(pool6, w['conv7'], pad=pad7)
        batch_norm7 = op.batch_normalize(conv7, w['bn7'])
        relu7 = op.relu(batch_norm7)
        pool7 = op.pool(relu7, p_h5, p_w5, stride5)

        '''   LAYER 8   '''
        conv8 = op.convolve(pool7, w['conv8'], pad=pad8)
        batch_norm8 = op.batch_normalize(conv8, w['bn8'])
        relu8 = op.relu(batch_norm8)

        '''   LAYER 9   '''
        conv9 = op.convolve(relu8, w['conv9'], pad=pad9)
        batch_norm9 = op.batch_normalize(conv9, w['bn9'])
        relu9 = op.relu(batch_norm9)

        '''   LAYER 10   '''
        conv10 = op.convolve(relu9, w['conv10'], pad=pad10)
        relu10 = op.relu(conv10)
        prediction = op.non_max_suppression(relu10, self.conv_thresh, self.IoU_thresh)

        return prediction

    def create_emotion_rec(self, infos, inp):
        pass

    def forward_prop(self, infos, inp, weights, training=False):
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

        #w = weights
        #mv = self.running_mean_var

        # normalize images
        inp = op.normalize(inp)

        '''   LAYER 1   '''
        conv1 = op.convolve(inp, self.w_conv1, pad=pad1)
        relu1 = op.sigmoid(conv1)
        batch_norm1 = op.batch_normalize(relu1, self.w_bnb1, self.w_bng1, mv['bn1'], training=True)

        '''   LAYER 2   '''
        conv2 = op.convolve(batch_norm1[0], self.w_conv2, pad=pad2)
        relu2 = op.sigmoid(conv2)
        batch_norm2 = op.batch_normalize(relu2, self.w_bnb2, self.w_bng2, mv['bn2'], training=True)
        pool2 = op.pool(batch_norm2[0], p_h1, p_w1, stride1)

        '''   LAYER 3   '''
        conv3 = op.convolve(pool2, self.w_conv3, pad=pad3)
        relu3 = op.sigmoid(conv3)
        batch_norm3 = op.batch_normalize(relu3, self.w_bnb3, self.w_bng3, mv['bn3'], training=True)

        '''   LAYER 4   '''
        conv4 = op.convolve(batch_norm3, w['conv4'], pad=pad4)
        relu4 = op.relu(conv4)
        # batch_norm4 = op.batch_normalize(relu4, self.w_bnb4, self.w_bng4, mv['bn4'], training=True)
        pool4 = op.pool(relu4, p_h2, p_w2, stride2)

        '''   LAYER 5   '''
        conv5 = op.convolve(pool4, self.w_conv5, pad=pad5)
        relu5 = op.relu(conv5)
        # batch_norm5 = op.batch_normalize(relu5, self.w_bnb5, self.w_bng5, mv['bn5'], training=True)
        pool5 = op.pool(relu5, p_h3, p_w3, stride3)

        '''   LAYER 6   '''
        conv6 = op.convolve(pool5, self.w_conv6, pad=pad6)
        relu6 = op.relu(conv6)
        # batch_norm6 = op.batch_normalize(relu6, self.w_bnb6, self.w_bng6, mv['bn6'], training=True)
        pool6 = op.pool(relu6, p_h4, p_w4, stride4)

        '''   LAYER 7   '''
        conv7 = op.convolve(pool6, self.w_conv7, pad=pad7)
        relu7 = op.relu(conv7)
        # batch_norm7 = op.batch_normalize(relu7, self.w_bnb7, self.w_bng7, mv['bn7'], training=True)
        pool7 = op.pool(relu7, p_h5, p_w5, stride5)

        """
        '''   LAYER 8   '''
        conv8 = op.convolve(pool7, self.w_conv8, pad=pad8)
        relu8 = op.relu(conv8)
        # batch_norm8 = op.batch_normalize(relu8, self.w_bnb8, self.w_bng8, mv['bn8'], training=True)

        '''   LAYER 9   '''
        conv9 = op.convolve(relu8, self.w_conv9, pad=pad9)
        relu9 = op.relu(conv9)
        # batch_norm9 = op.batch_normalize(relu9, self.w_bnb9, self.w_bng9, mv['bn9'], training=True)

        '''   LAYER 10   '''
        conv10 = op.convolve(relu9, self.w_conv10, pad=pad10)
        relu10 = op.relu(conv10)
        #prediction = op.non_max_suppression(relu10, self.conv_thresh, self.IoU_thresh)
        """

        '''   FULLY CONNECTED LAYER   '''
        N, D, H, W = batch_norm3[0].shape
        flatten = batch_norm3[0].reshape(-1, D * H * W)
        full_conn = op.full_conn(flatten, w['full'])
        #prediction = op.sigmoid(full_conn)
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
        weights = {}
        running_mean_var = {}

        n_fm1, f_h, f_w, _, _ = infos[0]
        self.w_conv1 = op.initialize_weights((n_fm1, 3, f_h, f_w), 'w_conv1')
        self.w_bnb1 = op.initialize_weights((n_fm1,), 'w_bnb1')
        self.w_bng1 = op.initialize_weights((n_fm1,), 'w_bng1')
        self.running_mean_var_bn1 = tf.zeros([1])

        n_fm2, f_h, f_w, _, _ = infos[1]
        self.w_conv2 = op.initialize_weights((n_fm2, n_fm1, f_h, f_w), 'w_conv2')
        self.w_bnb2 = op.initialize_weights((n_fm2,), 'w_bnb2')
        self.w_bng2 = op.initialize_weights((n_fm2,), 'w_bnb2')
        self.running_mean_var_bn2 = tf.zeros([1])

        n_fm3, f_h, f_w, _, _ = infos[3]
        self.w_conv3 = op.initialize_weights((n_fm3, n_fm2, f_h, f_w), 'w_conv3')
        self.w_bnb3 = op.initialize_weights((n_fm3,), 'w_bnb3')
        self.w_bng3 = op.initialize_weights((n_fm3,), 'w_bng3')
        self.running_mean_var_bn3 = tf.zeros([1])

        n_fm4, f_h, f_w, _, _ = infos[4]
        self.w_conv4 = op.initialize_weights((n_fm4, n_fm3, f_h, f_w), 'w_conv4')
        self.w_bnb4 = op.initialize_weights((2, n_fm4), 'w_bnb4')
        self.w_bng4 = op.initialize_weights((2, n_fm4), 'w_bng4')
        self.running_mean_var_bn4 = tf.zeros([1])

        n_fm5, f_h, f_w, _, _ = infos[6]
        self.w_conv5 = op.initialize_weights((n_fm5, n_fm4, f_h, f_w), 'w_conv5')
        self.w_bnb5 = op.initialize_weights((2, n_fm5), 'w_bnb5')
        self.w_bng5 = op.initialize_weights((2, n_fm5), 'w_bng5')
        self.running_mean_var_bn5 = tf.zeros([1])

        n_fm6, f_h, f_w, _, _ = infos[8]
        self.w_conv6 = op.initialize_weights((n_fm6, n_fm5, f_h, f_w), 'w_conv6')
        self.w_bnb6 = op.initialize_weights((2, n_fm6), 'w_bnb6')
        self.w_bng6 = op.initialize_weights((2, n_fm6), 'w_bng6')
        self.running_mean_var_bn6 = tf.zeros([1])

        n_fm7, f_h, f_w, _, _ = infos[10]
        self.w_conv7 = op.initialize_weights((n_fm7, n_fm6, f_h, f_w), 'w_conv7')
        self.w_bnb7 = op.initialize_weights((2, n_fm7), 'w_bnb7')
        self.w_bng7 = op.initialize_weights((2, n_fm7), 'w_bng7')
        self.running_mean_var_bn7 = tf.zeros([1])

        """
        n_fm8, f_h, f_w, _, _ = infos[12]
        self.w_conv8 = op.initialize_weights((n_fm8, n_fm7, f_h, f_w), 'w_conv8')
        self.w_bnb8 = op.initialize_weights((2, n_fm8), 'w_bnb8')
        self.w_bng8 = op.initialize_weights((2, n_fm8), 'w_bng8')
        self.running_mean_var_bn8 = tf.zeros([1])

        n_fm9, f_h, f_w, _, _ = infos[13]
        self.w_conv9 = op.initialize_weights((n_fm9, n_fm8, f_h, f_w), 'w_conv9')
        self.w_bnb9 = op.initialize_weights((2, n_fm9), 'w_bnb9')
        self.w_bng9 = op.initialize_weights((2, n_fm9), 'w_bng9')
        self.running_mean_var_bn9 = tf.zeros([1])

        n_fm10, f_h, f_w, _, _ = infos[14]
        self.w_conv10 = op.initialize_weights((n_fm10, n_fm9, f_h, f_w), 'w_conv10')
        """
        fc_in = 13 * 13 * 256
        self.w_full = op.initialize_weights((fc_in, 20), 'w_full')

    def update_weights(self, weights, gradients, learning_rate, batch_size):
        for w in weights:
            weights[w] = op.update_weights(weights[w], gradients[w], learning_rate, batch_size)
        return weights
