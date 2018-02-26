import tensorflow as tf
import numpy as np
from math import sqrt
import Operations as op

class Neural_Network:

    def __init__(self, model_type, infos, hypers, training=False):
        #model_type can specify which application we want to customize our operations for.

        if model_type == "facial_recognition":
            self.conv_thresh, self.IoU_thresh = hypers
            self._init_facial_rec_weights(infos)
            #self.create_facial_rec(infos)

        if model_type == "emotion_recognition":
            #self.create_emotion_rec(infos)
            pass

        self.batch_norm_avgs = np.zeros((hypers[0], 2))

    # Create the shell of the facial recognition neural network.
    def create_facial_rec(self, infos, inp):

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

        w = self.weights

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

        cache = 0
        if training:
            samp_batch_norms = np.array([w['bn1'], w['bn2'], w['bn3'], w['bn4'],
                                         w['bn5'], w['bn6'], w['bn7'], w['b8'], w['bn9']])

            self.batch_norm_avgs = 0.9 * self.batch_norm_avgs + 0.1 * samp_batch_norms

            cache = [conv10, batch_norm9, conv9, batch_norm8, conv8,
                     pool7, batch_norm7, conv7, pool6, batch_norm6, conv6,
                     pool5, batch_norm5, conv5, pool4, batch_norm4, conv4,
                     batch_norm3, conv3, pool2, batch_norm2, conv2,
                     batch_norm1, conv1, inp]
        return prediction, cache

    def backward_prop(self, cache, output, labels, infos):
        error = op.mse_prime(output, labels)

        w = self.weights

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

        '''   LAYER 10   '''
        relu_error = op.relu_prime(cache[0]) * error
        conv_dx, conv10_dw = op.convolve_backprop(relu_error, cache[1][0], w['conv10'], pad=pad10)
        w['conv10'] = op.update_weights(w['conv10'], conv10_dw)

        '''   LAYER 9   '''
        relu_error = op.relu_prime(cache[2]) * conv_dx
        batch_norm_dx, batch_norm9_db, batch_norm9_dg = op.batch_norm_backprop(relu_error, cache[1][1])
        conv_dx, conv9_dw = op.convolve(relu_error, cache[3][0], w['conv9'], pad=pad9)
        w['bn9'] = op.update_weights([w['bn9']], [batch_norm9_db, batch_norm9_dg])
        w['conv9'] = op.update_weights(w['conv9'], conv9_dw)

        '''   LAYER 8   '''
        relu_error = op.relu_prime(cache[4]) * conv_dx
        batch_norm_dx, batch_norm8_db, batch_norm8_dg = op.batch_norm_backprop(relu_error, cache[3][1])
        conv_dx, conv8_dw = op.convolve(relu_error, cache[5], w['conv8'], pad=pad8)
        w['bn8'] = op.update_weights([w['bn8']], [batch_norm8_db, batch_norm8_dg])
        w['conv8'] = op.update_weights(w['conv8'], conv8_dw)

        '''   LAYER 7   '''
        pool = op.pool_backprop(conv_dx, cache[6][0], p_h5, p_w5, stride5)
        relu_error = op.relu_prime(cache[7]) * pool
        batch_norm_dx, batch_norm7_db, batch_norm7_dg = op.batch_norm_backprop(relu_error, cache[6][1])
        conv_dx, conv7_dw = op.convolve(relu_error, cache[8], w['conv7'], pad=pad7)
        w['bn7'] = op.update_weights([w['bn7']], [batch_norm7_db, batch_norm7_dg])
        w['conv7'] = op.update_weights(w['conv7'], conv7_dw)

        '''   LAYER 6   '''
        pool = op.pool_backprop(conv_dx, cache[9][0], p_h4, p_w4, stride4)
        relu_error = op.relu_prime(cache[10]) * pool
        batch_norm_dx, batch_norm6_db, batch_norm6_dg = op.batch_norm_backprop(relu_error, cache[9][1])
        conv_dx, conv6_dw = op.convolve(relu_error, cache[11], w['conv6'], pad=pad6)
        w['bn6'] = op.update_weights(w['bn6'], [batch_norm6_db, batch_norm6_dg])
        w['conv6'] = op.update_weights(w['conv6'], conv6_dw)

        '''   LAYER 5   '''
        pool = op.pool_backprop(conv_dx, cache[12][0], p_h3, p_w3, stride3)
        relu_error = op.relu_prime(cache[13]) * pool
        batch_norm_dx, batch_norm5_db, batch_norm5_dg = op.batch_norm_backprop(relu_error, cache[12][1])
        conv_dx, conv5_dw = op.convolve(relu_error, cache[14], w['conv5'], pad=pad5)
        w['bn5'] = op.update_weights([w['bn5']], [batch_norm5_db, batch_norm5_dg])
        w['conv5'] = op.update_weights(w['conv5'], conv5_dw)

        '''   LAYER 4   '''
        pool = op.pool_backprop(conv_dx, cache[15][0], p_h2, p_w2, stride2)
        relu_error = op.relu_prime(cache[16]) * pool
        batch_norm_dx, batch_norm4_db, batch_norm4_dg = op.batch_norm_backprop(relu_error, cache[15][1])
        conv_dx, conv4_dw = op.convolve(relu_error, cache[17][0], w['conv4'], pad=pad4)
        w['bn4'] = op.update_weights([w['bn4']], [batch_norm4_db, batch_norm4_dg])
        w['conv4'] = op.update_weights(w['conv4'], conv4_dw)

        '''   LAYER 3   '''
        relu_error = op.relu_prime(cache[18]) * conv_dx
        batch_norm_dx, batch_norm3_db, batch_norm3_dg = op.batch_norm_backprop(relu_error, cache[17][1])
        conv_dx, conv3_dw = op.convolve(relu_error, cache[19], w['conv3'], pad=pad3)
        w['bn3'] = op.update_weights([w['bn3']], [batch_norm3_db, batch_norm3_dg])
        w['conv3'] = op.update_weights(w['conv3'], conv3_dw)

        '''   LAYER 2   '''
        pool = op.pool_backprop(conv_dx, cache[20][0], p_h1, p_w1, stride1)
        relu_error = op.relu_prime(cache[21]) * pool
        batch_norm_dx, batch_norm2_db, batch_norm2_dg = op.batch_norm_backprop(relu_error, cache[20][1])
        conv_dx, conv2_dw = op.convolve(relu_error, cache[22][0], w['conv2'], pad=pad2)
        w['bn2'] = op.update_weights([w['bn2']], [batch_norm2_db, batch_norm2_dg])
        w['conv2'] = op.update_weights(w['conv2'], conv2_dw)

        '''   LAYER 1   '''
        relu_error = op.relu_prime(cache[23]) * conv_dx
        batch_norm_dx, batch_norm1_db, batch_norm1_dg = op.batch_norm_backprop(relu_error, cache[22][1])
        _, conv1_dw = op.convolve(relu_error, cache[24], w['conv1'], pad=pad1)
        w['bn1'] = op.update_weights([w['bn1']], [batch_norm1_db, batch_norm1_dg])
        w['conv1'] = op.update_weights(w['conv1'], conv1_dw)

        gradients = {'conv1': conv1_dw, 'conv2': conv2_dw, 'conv3': conv3_dw, 'conv4': conv4_dw, 'conv5': conv5_dw,
                     'conv6': conv6_dw, 'conv7': conv7_dw, 'conv8': conv8_dw, 'conv9': conv9_dw, 'conv10': conv10_dw,
                     'bn1': [batch_norm1_db, batch_norm1_dg], 'bn2': [batch_norm2_db, batch_norm2_dg],
                     'bn3': [batch_norm3_db, batch_norm3_dg], 'bn4': [batch_norm4_db, batch_norm4_dg],
                     'bn5': [batch_norm5_db, batch_norm5_dg], 'bn6': [batch_norm6_db, batch_norm6_dg],
                     'bn7': [batch_norm7_db, batch_norm7_dg], 'bn8': [batch_norm8_db, batch_norm8_dg],
                     'bn9': [batch_norm9_db, batch_norm9_dg]}

        return gradients

    def _init_facial_rec_weights(self, infos):
        weights = {}

        n_fm1, f_h, f_w, _, _ = infos[0]
        weights['conv1'] = op.initialize_weights([n_fm1, 3, f_h, f_w])

        n_fm2, f_h, f_w, _, _ = infos[1]
        weights['conv2'] = op.initialize_weights([n_fm2, n_fm1, f_h, f_w])

        n_fm3, f_h, f_w, _, _ = infos[3]
        weights['conv3'] = op.initialize_weights([n_fm3, n_fm2, f_h, f_w])

        n_fm4, f_h, f_w, _, _ = infos[4]
        weights['conv4'] = op.initialize_weights([n_fm4, n_fm3, f_h, f_w])

        n_fm5, f_h, f_w, _, _ = infos[6]
        weights['conv5'] = op.initialize_weights([n_fm5, n_fm4, f_h, f_w])

        n_fm6, f_h, f_w, _, _ = infos[8]
        weights['conv6'] = op.initialize_weights([n_fm6, n_fm5, f_h, f_w])

        n_fm7, f_h, f_w, _, _ = infos[10]
        weights['conv7'] = op.initialize_weights([n_fm7, n_fm6, f_h, f_w])

        n_fm8, f_h, f_w, _, _ = infos[12]
        weights['conv8'] = op.initialize_weights([n_fm8, n_fm7, f_h, f_w])

        n_fm9, f_h, f_w, _, _ = infos[13]
        weights['conv9'] = op.initialize_weights([n_fm9, n_fm8, f_h, f_w])

        n_fm10, f_h, f_w, _, _ = infos[14]
        weights['conv10'] = op.initialize_weights([n_fm10, n_fm9, f_h, f_w])

        for i in range(9):
            weights["bn" + str(i+1)] = op.initialize_weights([2])

        self.weights = weights
