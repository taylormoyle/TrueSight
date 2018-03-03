import tensorflow as tf
import numpy as np
from math import sqrt
import Operations as op

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

        w = weights
        #mv = self.batch_MV

        # normalize images
        inp = op.normalize(inp)

        '''   LAYER 1   '''
        conv1 = op.convolve(inp, w['conv1'], pad=pad1)
        relu1 = op.sigmoid(conv1)
        # batch_norm1 = op.batch_normalize(relu1, w['bn1'], mv['bn1'], training=True)

        '''   LAYER 2   '''
        conv2 = op.convolve(relu1, w['conv2'], pad=pad2)
        relu2 = op.sigmoid(conv2)
        # batch_norm2 = op.batch_normalize(relu2, w['bn2'], mv['bn2'], training=True)
        pool2 = op.pool(relu2, p_h1, p_w1, stride1)

        '''   LAYER 3   '''
        conv3 = op.convolve(pool2, w['conv3'], pad=pad3)
        relu3 = op.sigmoid(conv3)
        # batch_norm3 = op.batch_normalize(relu3, w['bn3'], mv['bn3'], training=True)

        '''   LAYER 4   '''
        conv4 = op.convolve(relu3, w['conv4'], pad=pad4)
        relu4 = op.relu(conv4)
        # batch_norm4 = op.batch_normalize(relu4, w['bn4'], mv['bn4'], training=True)
        pool4 = op.pool(relu4, p_h2, p_w2, stride2)
        """
        '''   LAYER 5   '''
        conv5 = op.convolve(pool4, w['conv5'], pad=pad5)
        relu5 = op.relu(conv5)
        # batch_norm5 = op.batch_normalize(relu5, w['bn5'], mv['bn5'], training=True)
        pool5 = op.pool(relu5, p_h3, p_w3, stride3)

        '''   LAYER 6   '''
        conv6 = op.convolve(pool5, w['conv6'], pad=pad6)
        relu6 = op.relu(conv6)
        # batch_norm6 = op.batch_normalize(relu6, w['bn6'], mv['bn6'], training=True)
        pool6 = op.pool(relu6, p_h4, p_w4, stride4)

        '''   LAYER 7   '''
        conv7 = op.convolve(pool6, w['conv7'], pad=pad7)
        relu7 = op.relu(conv7)
        # batch_norm7 = op.batch_normalize(relu7, w['bn7'], mv['bn7'], training=True)
        pool7 = op.pool(relu7, p_h5, p_w5, stride5)

        '''   LAYER 8   '''
        conv8 = op.convolve(pool7, w['conv8'], pad=pad8)
        relu8 = op.relu(conv8)
        # batch_norm8 = op.batch_normalize(relu8, w['bn8'], mv['bn8'], training=True)

        '''   LAYER 9   '''
        conv9 = op.convolve(relu8, w['conv9'], pad=pad9)
        relu9 = op.relu(conv9)
        # batch_norm9 = op.batch_normalize(relu9, w['bn9'], mv['bn9'], training=True)

        '''   LAYER 10   '''
        conv10 = op.convolve(relu9, w['conv10'], pad=pad10)
        relu10 = op.relu(conv10)
        #prediction = op.non_max_suppression(relu10, self.conv_thresh, self.IoU_thresh)
        """
        '''   FULLY CONNECTED LAYER   '''
        N, D, H, W = tf.shape(pool4)
        flatten = tf.reshape(pool4, [-1, (D * H * W)])
        full_conn = op.full_conn(flatten, w['full'])
        prediction = op.sigmoid(full_conn)

        if training:
            '''
            if len(mv['bn1']) < 1:
                mv['bn1'] = batch_norm1[2]
                #mv['bn7'] = batch_norm7[2]
                #mv['bn8'] = batch_norm8[2]
                mv['bn2'] = batch_norm2[2]
                #mv['bn3'] = batch_norm3[2]
                #mv['bn4'] = batch_norm4[2]
                #mv['bn5'] = batch_norm5[2]
                #mv['bn6'] = batch_norm6[2]
                #mv['bn8'] = batch_norm9[2]
            else:
                 mv['bn1'] = 0.9 * mv['bn1'] + 0.1 * batch_norm1[2]
                 mv['bn2'] = 0.9 * mv['bn2'] + 0.1 * batch_norm2[2]
                 #mv['bn3'] = 0.9 * mv['bn3'] + 0.1 * batch_norm3[2]
                 #mv['bn4'] = 0.9 * mv['bn4'] + 0.1 * batch_norm4[2]
                 #mv['bn5'] = 0.9 * mv['bn5'] + 0.1 * batch_norm5[2]
                 #mv['bn6'] = 0.9 * mv['bn6'] + 0.1 * batch_norm6[2]
                 #mv['bn7'] = 0.9 * mv['bn7'] + 0.1 * batch_norm7[2]
                 #mv['bn8'] = 0.9 * mv['bn8'] + 0.1 * batch_norm8[2]
                 #mv['bn9'] = 0.9 * mv['bn9'] + 0.1 * batch_norm9[2]
            '''

            cache = {#'c10': conv10, 'c9': conv9, 'c8': conv8, 'c7': conv7, 'c6': conv6, 'c5': conv5, 'c4': conv4,
                     #'c3': conv3, 'c2': conv2, 'c1': conv1, 'r10': relu10, 'r9': relu9, 'r8': relu8, 'r7': relu7,
                     #'r6': relu6, 'r5': relu5, 'r4': relu4, 'r3': relu3, 'r2': relu2, 'r1': relu1, 'p7': pool7,
                     #'p6': pool6, 'p5': pool5,
                     'r4': relu4, 'r3': relu3, 'r2': relu2, 'r1': relu1, 'c3': conv3, 'c2': conv2, 'c1': conv1, 'c4': conv4, 'p4': pool4, 'p2': pool2, 'inp': inp, 'fc': full_conn, 'fl': flatten}

            return prediction, cache
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

        batch_size, l = tf.shape(output)
        error = op.mse_prime(output, labels)
        delta_out = error * op.sigmoid_prime(cache['fc']) / batch_size

        '''   FULLY CONNECTED LAYER   '''
        full_dx, full_dw = op.full_conn_backprop(delta_out, cache['fl'], w['full'])
        full_dx = tf.reshape(full_dx, tf.shape(cache['p4']))

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
        """
        '''   LAYER 4   '''
        pool = op.pool_backprop(full_dx, cache['r4'], p_h2, p_w2, stride2)
        #batch_norm_dx, batch_norm4_dbg = op.batch_norm_backprop(pool, cache[12][1])
        relu_error = op.relu_prime(cache['c4']) * pool #* batch_norm_dx
        conv_dx, conv4_dw = op.convolve_backprop(relu_error, cache['r3'], w['conv4'], pad=pad4)

        '''   LAYER 3   '''
        #batch_norm_dx, batch_norm3_dbg = op.batch_norm_backprop(conv_dx, cache[14][1])
        relu_error = op.sigmoid_prime(cache['c3']) * conv_dx #full_dx
        conv_dx, conv3_dw = op.convolve_backprop(relu_error, cache['p2'], w['conv3'], pad=pad3)

        '''   LAYER 2   '''
        pool = op.pool_backprop(conv_dx, cache['r2'], p_h1, p_w1, stride1)
        #batch_norm_dx, batch_norm2_dbg = op.batch_norm_backprop(pool, cache[4][1])
        relu_error = op.sigmoid_prime(cache['c2']) * pool #* batch_norm_dx
        conv_dx, conv2_dw = op.convolve_backprop(relu_error, cache['r1'], w['conv2'], pad=pad2)

        '''   LAYER 1   '''
        #batch_norm_dx, batch_norm1_dbg = op.batch_norm_backprop(conv_dx, cache[6][1])
        relu_error = op.sigmoid_prime(cache['c1']) * conv_dx #* batch_norm_dx
        _, conv1_dw = op.convolve_backprop(relu_error, cache['inp'], w['conv1'], pad=pad1)

        gradients = {'full': full_dw,
                     'conv1': conv1_dw, 'conv2': conv2_dw, 'conv3': conv3_dw, 'conv4': conv4_dw} #, 'conv5': conv5_dw,
                     #'conv6': conv6_dw, 'conv7': conv7_dw, 'conv8': conv8_dw, 'conv9': conv9_dw, 'conv10': conv10_dw}

        return gradients

    def init_facial_rec_weights(self, infos):
        weights = {}
        #batch_MV = {}

        n_fm1, f_h, f_w, _, _ = infos[0]
        weights['conv1'] = op.initialize_weights((n_fm1, 3, f_h, f_w))
        #weights['bn1'] = op.initialize_weights((2, n_fm1))
        #batch_MV['bn1'] = []

        n_fm2, f_h, f_w, _, _ = infos[1]
        weights['conv2'] = op.initialize_weights((n_fm2, n_fm1, f_h, f_w))
        #weights['bn2'] = op.initialize_weights((2, n_fm2))
        #batch_MV['bn2'] = []

        n_fm3, f_h, f_w, _, _ = infos[3]
        weights['conv3'] = op.initialize_weights((n_fm3, n_fm2, f_h, f_w))
        #weights['bn3'] = op.initialize_weights((2, n_fm3))
        #batch_MV['bn3'] = []

        n_fm4, f_h, f_w, _, _ = infos[4]
        weights['conv4'] = op.initialize_weights((n_fm4, n_fm3, f_h, f_w))
        #weights['bn4'] = op.initialize_weights((2, n_fm4))
        #batch_MV['bn4'] = []
        """
        n_fm5, f_h, f_w, _, _ = infos[6]
        weights['conv5'] = op.initialize_weights((n_fm5, n_fm4, f_h, f_w))
        #weights['bn5'] = op.initialize_weights((2, n_fm5))
        #batch_MV['bn5'] = []

        n_fm6, f_h, f_w, _, _ = infos[8]
        weights['conv6'] = op.initialize_weights((n_fm6, n_fm5, f_h, f_w))
        #weights['bn6'] = op.initialize_weights((2, n_fm6))
        #batch_MV['bn6'] = []

        n_fm7, f_h, f_w, _, _ = infos[10]
        weights['conv7'] = op.initialize_weights((n_fm7, n_fm6, f_h, f_w))
        #weights['bn7'] = op.initialize_weights((2, n_fm7))
        #batch_MV['bn7'] = []

        n_fm8, f_h, f_w, _, _ = infos[12]
        weights['conv8'] = op.initialize_weights((n_fm8, n_fm7, f_h, f_w))
        #weights['bn8'] = op.initialize_weights((2, n_fm8))
        #batch_MV['bn8'] = []

        n_fm9, f_h, f_w, _, _ = infos[13]
        weights['conv9'] = op.initialize_weights((n_fm9, n_fm8, f_h, f_w))
        #weights['bn9'] = op.initialize_weights((2, n_fm9))
        #batch_MV['bn9'] = []

        n_fm10, f_h, f_w, _, _ = infos[14]
        weights['conv10'] = op.initialize_weights((n_fm10, n_fm9, f_h, f_w))
        """
        fc_in = 104*104*n_fm4
        weights['full'] = op.initialize_weights((fc_in, 20))

        #self.batch_MV = batch_MV
        return weights

    def update_weights(self, weights, gradients, learning_rate, batch_size):
        for w in weights:
            weights[w] = op.update_weights(weights[w], gradients[w], learning_rate, batch_size)
        return weights
