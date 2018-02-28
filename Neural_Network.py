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
        mv = self.batch_MV

        # normalize images
        #inp = op.normalize(inp)

        '''   LAYER 1   '''
        conv1 = op.convolve(inp, w['conv1'], pad=pad1)
        relu1 = op.sigmoid(conv1)
        batch_norm1 = op.batch_normalize(relu1, w['bn1'], mv['bn1'], training=True)

        '''   LAYER 2   '''
        conv2 = op.convolve(batch_norm1[0], w['conv2'], pad=pad2)
        relu2 = op.sigmoid(conv2)
        batch_norm2 = op.batch_normalize(relu2, w['bn2'], mv['bn2'], training=True)
        pool2 = op.pool(batch_norm2[0], p_h1, p_w1, stride1)

        '''   LAYER 3   '''
        conv3 = op.convolve(pool2, w['conv3'], pad=pad3)
        relu3 = op.sigmoid(conv3)

        """
        batch_norm3 = op.batch_normalize(relu3, w['bn3'], mv['bn3'], training=True)

        '''   LAYER 4   '''
        conv4 = op.convolve(batch_norm3[0], w['conv4'], pad=pad4)
        relu4 = op.relu(conv4)
        batch_norm4 = op.batch_normalize(relu4, w['bn4'], mv['bn4'], training=True)
        pool4 = op.pool(batch_norm4[0], p_h2, p_w2, stride2)

        '''   LAYER 5   '''
        conv5 = op.convolve(pool4, w['conv5'], pad=pad5)
        relu5 = op.relu(conv5)
        batch_norm5 = op.batch_normalize(relu5, w['bn5'], mv['bn5'], training=True)
        pool5 = op.pool(batch_norm5[0], p_h3, p_w3, stride3)

        '''   LAYER 6   '''
        conv6 = op.convolve(pool5, w['conv6'], pad=pad6)
        relu6 = op.relu(conv6)
        batch_norm6 = op.batch_normalize(relu6, w['bn6'], mv['bn6'], training=True)
        pool6 = op.pool(batch_norm6[0], p_h4, p_w4, stride4)

        '''   LAYER 7   '''
        conv7 = op.convolve(pool6, w['conv7'], pad=pad7)
        relu7 = op.relu(conv7)
        batch_norm7 = op.batch_normalize(relu7, w['bn7'], mv['bn7'], training=True)
        pool7 = op.pool(batch_norm7[0], p_h5, p_w5, stride5)
        
        '''   LAYER 8   '''
        conv8 = op.convolve(pool7, w['conv8'], pad=pad8)
        relu8 = op.relu(conv8)
        batch_norm8 = op.batch_normalize(relu8, w['bn8'], mv['bn8'], training=True)

        '''   LAYER 9   '''
        conv9 = op.convolve(batch_norm8[0], w['conv9'], pad=pad9)
        relu9 = op.relu(conv9)
        batch_norm9 = op.batch_normalize(relu9, w['bn9'], mv['bn9'], training=True)

        '''   LAYER 10   '''
        conv10 = op.convolve(batch_norm9[0], w['conv10'], pad=pad10)
        relu10 = op.relu(conv10)
        #prediction = op.non_max_suppression(relu10, self.conv_thresh, self.IoU_thresh)
        """

        '''   FULLY CONNECTED LAYER   '''
        N, D, H, W = relu3.shape
        flatten = relu3.reshape(-1, D*H*W)
        full_conn = op.full_conn(flatten, w['full'])
        prediction = op.sigmoid(full_conn)

        if training:
            if len(mv['bn1']) < 1:
                mv['bn1'] = batch_norm1[2]
                mv['bn2'] = batch_norm2[2]
                #mv['bn3'] = batch_norm3[2]
                #mv['bn4'] = batch_norm4[2]
                #mv['bn5'] = batch_norm5[2]
                #mv['bn6'] = batch_norm6[2]
                #mv['bn7'] = batch_norm7[2]
                #mv['bn8'] = batch_norm8[2]
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

            cache = [#conv10, batch_norm9, conv9, batch_norm8, conv8,
                     full_conn, flatten,
                     #pool7, batch_norm7, conv7, pool6, batch_norm6, conv6,
                     #pool5, batch_norm5, conv5, pool4, batch_norm4, conv4, batch_norm3,
                     conv3, pool2, batch_norm2, conv2,
                     batch_norm1, conv1, inp]
            return prediction, cache
        return prediction

    def backward_prop(self, cache, output, labels, weights, infos, learning_rate):
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

        batch_size, l = output.shape
        error = op.mse_prime(output, labels)
        delta_out = error * op.sigmoid_prime(cache[0]) / batch_size

        '''   FULLY CONNECTED LAYER   '''
        full_dx, full_dw = op.full_conn_backprop(delta_out, cache[1], w['full'])
        #w['full'] = op.update_weights(w['full'], full_dw, learning_rate, batch_size)
        full_dx = full_dx.reshape(cache[2].shape)

        """
        '''   LAYER 10   '''
        relu_error = op.relu_prime(cache[0]) * error
        conv_dx, conv10_dw = op.convolve_backprop(relu_error, cache[1][0], w['conv10'], pad=pad10)
        w['conv10'] = op.update_weights(w['conv10'], conv10_dw, learning_rate, batch_size)

        '''   LAYER 9   '''
        batch_norm_dx, batch_norm9_dbg = op.batch_norm_backprop(conv_dx, cache[1][1])
        relu_error = op.relu_prime(cache[2]) * batch_norm_dx
        conv_dx, conv9_dw = op.convolve_backprop(relu_error, cache[3][0], w['conv9'], pad=pad9)
        w['bn9'] = op.update_weights(w['bn9'], batch_norm9_dbg, learning_rate, batch_size)
        w['conv9'] = op.update_weights(w['conv9'], conv9_dw, learning_rate, batch_size)

        '''   LAYER 8   '''
        batch_norm_dx, batch_norm8_dbg = op.batch_norm_backprop(conv_dx, cache[3][1])
        relu_error = op.relu_prime(cache[4]) * batch_norm_dx
        conv_dx, conv8_dw = op.convolve_backprop(relu_error, cache[5], w['conv8'], pad=pad8)
        w['bn8'] = op.update_weights(w['bn8'], batch_norm8_dbg, learning_rate, batch_size)
        w['conv8'] = op.update_weights(w['conv8'], conv8_dw, learning_rate, batch_size)

        '''   LAYER 7   '''
        pool = op.pool_backprop(full_dx, cache[3][0], p_h5, p_w5, stride5)
        batch_norm_dx, batch_norm7_dbg = op.batch_norm_backprop(pool, cache[3][1])
        relu_error = op.relu_prime(cache[4]) * batch_norm_dx
        conv_dx, conv7_dw = op.convolve_backprop(relu_error, cache[5], w['conv7'], pad=pad7)
        w['bn7'] = op.update_weights(w['bn7'], batch_norm7_dbg, learning_rate, batch_size)
        w['conv7'] = op.update_weights(w['conv7'], conv7_dw, learning_rate, batch_size)

        '''   LAYER 6   '''
        pool = op.pool_backprop(conv_dx, cache[6][0], p_h4, p_w4, stride4)
        batch_norm_dx, batch_norm6_dbg = op.batch_norm_backprop(pool, cache[6][1])
        relu_error = op.relu_prime(cache[7]) * batch_norm_dx
        conv_dx, conv6_dw = op.convolve_backprop(relu_error, cache[8], w['conv6'], pad=pad6)
        w['bn6'] = op.update_weights(w['bn6'], batch_norm6_dbg, learning_rate, batch_size)
        w['conv6'] = op.update_weights(w['conv6'], conv6_dw, learning_rate, batch_size)

        '''   LAYER 5   '''
        pool = op.pool_backprop(conv_dx, cache[9][0], p_h3, p_w3, stride3)
        batch_norm_dx, batch_norm5_dbg = op.batch_norm_backprop(pool, cache[9][1])
        relu_error = op.relu_prime(cache[10]) * batch_norm_dx
        conv_dx, conv5_dw = op.convolve_backprop(relu_error, cache[11], w['conv5'], pad=pad5)
        w['bn5'] = op.update_weights(w['bn5'], batch_norm5_dbg, learning_rate, batch_size)
        w['conv5'] = op.update_weights(w['conv5'], conv5_dw, learning_rate, batch_size)

        '''   LAYER 4   '''
        pool = op.pool_backprop(conv_dx, cache[12][0], p_h2, p_w2, stride2)
        batch_norm_dx, batch_norm4_dbg = op.batch_norm_backprop(pool, cache[12][1])
        relu_error = op.relu_prime(cache[13]) * batch_norm_dx
        conv_dx, conv4_dw = op.convolve_backprop(relu_error, cache[14][0], w['conv4'], pad=pad4)
        w['bn4'] = op.update_weights(w['bn4'], batch_norm4_dbg, learning_rate, batch_size)
        w['conv4'] = op.update_weights(w['conv4'], conv4_dw, learning_rate, batch_size)

        '''   LAYER 3   '''
        batch_norm_dx, batch_norm3_dbg = op.batch_norm_backprop(conv_dx, cache[14][1])
        """

        relu_error = op.sigmoid_prime(cache[2]) * full_dx
        conv_dx, conv3_dw = op.convolve_backprop(relu_error, cache[3], w['conv3'], pad=pad3)
        #w['bn3'] = op.update_weights(w['bn3'], batch_norm3_dbg, learning_rate, batch_size)
        #w['conv3'] = op.update_weights(w['conv3'], conv3_dw, learning_rate, batch_size)

        '''   LAYER 2   '''
        pool = op.pool_backprop(conv_dx, cache[4][0], p_h1, p_w1, stride1)
        batch_norm_dx, batch_norm2_dbg = op.batch_norm_backprop(pool, cache[4][1])
        relu_error = op.sigmoid_prime(cache[5]) * batch_norm_dx
        conv_dx, conv2_dw = op.convolve_backprop(relu_error, cache[6][0], w['conv2'], pad=pad2)
       # w['bn2'] = op.update_weights(w['bn2'], batch_norm2_dbg, learning_rate, batch_size)
        #w['conv2'] = op.update_weights(w['conv2'], conv2_dw, learning_rate, batch_size)

        '''   LAYER 1   '''
        batch_norm_dx, batch_norm1_dbg = op.batch_norm_backprop(conv_dx, cache[6][1])
        relu_error = op.sigmoid_prime(cache[7]) * batch_norm_dx
        _, conv1_dw = op.convolve_backprop(relu_error, cache[8], w['conv1'], pad=pad1)
        #w['bn1'] = op.update_weights(w['bn1'], batch_norm1_dbg, learning_rate, batch_size)
        #w['conv1'] = op.update_weights(w['conv1'], conv1_dw, learning_rate, batch_size)

        gradients = {'full': full_dw,
                     'conv1': conv1_dw, 'conv2': conv2_dw, 'conv3': conv3_dw,# 'conv4': conv4_dw, 'conv5': conv5_dw,
                     #'conv6': conv6_dw, 'conv7': conv7_dw, #'conv8': conv8_dw, 'conv9': conv9_dw, 'conv10': conv10_dw,
                     'bn1': batch_norm1_dbg, 'bn2': batch_norm2_dbg}#, 'bn3': batch_norm3_dbg, 'bn4': batch_norm4_dbg,
                     #'bn5': batch_norm5_dbg, 'bn6': batch_norm6_dbg, 'bn7': batch_norm7_dbg} #'bn8': batch_norm8_dbg,
                     #'bn9': batch_norm9_dbg}

        return gradients

    def init_facial_rec_weights(self, infos):
        weights = {}
        batch_MV = {}

        n_fm1, f_h, f_w, _, _ = infos[0]
        weights['conv1'] = op.initialize_weights((n_fm1, 3, f_h, f_w))
        weights['bn1'] = op.initialize_weights((2, n_fm1))
        batch_MV['bn1'] = []

        n_fm2, f_h, f_w, _, _ = infos[1]
        weights['conv2'] = op.initialize_weights((n_fm2, n_fm1, f_h, f_w))
        weights['bn2'] = op.initialize_weights((2, n_fm2))
        batch_MV['bn2'] = []

        n_fm3, f_h, f_w, _, _ = infos[3]
        weights['conv3'] = op.initialize_weights((n_fm3, n_fm2, f_h, f_w))

        """
        weights['bn3'] = op.initialize_weights((2, n_fm3))
        batch_MV['bn3'] = []

        n_fm4, f_h, f_w, _, _ = infos[4]
        weights['conv4'] = op.initialize_weights((n_fm4, n_fm3, f_h, f_w))
        weights['bn4'] = op.initialize_weights((2, n_fm4))
        batch_MV['bn4'] = []

        n_fm5, f_h, f_w, _, _ = infos[6]
        weights['conv5'] = op.initialize_weights((n_fm5, n_fm4, f_h, f_w))
        weights['bn5'] = op.initialize_weights((2, n_fm5))
        batch_MV['bn5'] = []

        n_fm6, f_h, f_w, _, _ = infos[8]
        weights['conv6'] = op.initialize_weights((n_fm6, n_fm5, f_h, f_w))
        weights['bn6'] = op.initialize_weights((2, n_fm6))
        batch_MV['bn6'] = []

        n_fm7, f_h, f_w, _, _ = infos[10]
        weights['conv7'] = op.initialize_weights((n_fm7, n_fm6, f_h, f_w))
        weights['bn7'] = op.initialize_weights((2, n_fm7))
        batch_MV['bn7'] = []

        n_fm8, f_h, f_w, _, _ = infos[12]
        weights['conv8'] = op.initialize_weights((n_fm8, n_fm7, f_h, f_w))
        weights['bn8'] = op.initialize_weights((2, n_fm8))
        batch_MV['bn8'] = []

        n_fm9, f_h, f_w, _, _ = infos[13]
        weights['conv9'] = op.initialize_weights((n_fm9, n_fm8, f_h, f_w))
        weights['bn9'] = op.initialize_weights((2, n_fm9))
        batch_MV['bn9'] = []

        n_fm10, f_h, f_w, _, _ = infos[14]
        weights['conv10'] = op.initialize_weights((n_fm10, n_fm9, f_h, f_w))
        """

        fc_in = 208*208*1
        weights['full'] = op.initialize_weights((fc_in, 20))

        self.batch_MV = batch_MV
        return weights
