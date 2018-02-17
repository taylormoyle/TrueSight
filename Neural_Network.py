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
            self.create_facial_rec(infos)

        if model_type == "emotion_recognition":
            self.create_emotion_rec()

    # Create the shell of the facial recognition neural network.
    def create_facial_rec(self, infos, inp):
        '''   WEIGHTS   '''

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

        '''   LAYER 1   '''
        conv1 = op.convole(inp, self.conv1_weights, pad=pad1)
        relu1 = op.relu(conv1)
        batch_norm1 = op.batch_normalize(relu1, self.gamma1, self.beta1)

        '''   LAYER 2   '''
        conv2 = op.convole(batch_norm1, self.conv2_weights, pad=pad2)
        relu2 = op.relu(conv2)
        batch_norm2 = op.batch_normalize(relu2, self.gamma2, self.beta2)
        pool2 = op.pool(batch_norm2, p_h1, p_w1, stride1)

        '''   LAYER 3   '''
        conv3 = op.convole(pool2, self.conv3_weights, pad=pad3)
        relu3 = op.relu(conv3)
        batch_norm3 = op.batch_normalize(relu3, self.gamma3, self.beta3)

        '''   LAYER 4   '''
        conv4 = op.convole(batch_norm3, self.conv4_weights, pad=pad4)
        relu4 = op.relu(conv4)
        batch_norm4 = op.batch_normalize(relu4, self.gamma4, self.beta4)
        pool4 = op.pool(batch_norm4, p_h2, p_w2, stride2)

        '''   LAYER 5   '''
        conv5 = op.convole(pool4, self.conv5_weights, pad=pad5)
        relu5 = op.relu(conv5)
        batch_norm5 = op.batch_normalize(relu5, self.gamma5, self.beta5)
        pool5 = op.pool(batch_norm5, p_h3, p_w3, stride3)

        '''   LAYER 6   '''
        conv6 = op.convole(pool5, self.conv6_weights, pad=pad6)
        relu6 = op.relu(conv6)
        batch_norm6 = op.batch_normalize(relu6, self.gamma6, self.beta6)
        pool6 = op.pool(batch_norm6, p_h4, p_w4, stride4)

        '''   LAYER 7   '''
        conv7 = op.convole(pool6, self.conv7_weights, pad=pad7)
        relu7 = op.relu(conv7)
        batch_norm7 = op.batch_normalize(relu7, self.gamma7, self.beta7)
        pool7 = op.pool(batch_norm7, p_h5, p_w5, stride5)

        '''   LAYER 8   '''
        conv8 = op.convole(pool7, self.conv8_weights, pad=pad8)
        relu8 = op.relu(conv8)
        batch_norm8 = op.batch_normalize(relu8, self.gamma8, self.beta8)

        '''   LAYER 9   '''
        conv9 = op.convole(batch_norm8, self.conv9_weights, pad=pad9)
        relu9 = op.relu(conv9)
        batch_norm9 = op.batch_normalize(relu9, self.gamma9, self.beta9)

        '''   LAYER 10   '''
        conv10 = op.convole(batch_norm9, self.conv10_weights, pad=pad10)
        relu10 = op.relu(conv10)
        prediction = op.non_max_suppression(relu10, self.conv_thresh, self.IoU_thresh)

        return prediction

    def create_emotion_rec(self, infos, inp):
        pass

    def forward_prop(self, inp):
        pass

    def backward_prop(self, inp):
        pass

    def _init_facial_rec_weights(self, infos):
        n_fm, f_h, f_w, _, _ = infos[0]
        self.conv1_weights = op.initialize_weights([n_fm, 3, f_h, f_w])

        n_fm, f_h, f_w, _, _ = infos[1]
        self.conv2_weights = op.initialize_weights([n_fm, self.conv1_weights.shape[0], f_h, f_w])

        n_fm, f_h, f_w, _, _ = infos[3]
        self.conv3_weights = op.initialize_weights([n_fm, self.conv2_weights.shape[0], f_h, f_w])

        n_fm, f_h, f_w, _, _ = infos[4]
        self.conv4_weights = op.initialize_weights([n_fm, self.conv3_weights.shape[0], f_h, f_w])

        n_fm, f_h, f_w, _, _ = infos[6]
        self.conv5_weights = op.initialize_weights([n_fm, self.conv4_weights.shape[0], f_h, f_w])

        n_fm, f_h, f_w, _, _ = infos[8]
        self.conv6_weights = op.initialize_weights([n_fm, self.conv5_weights.shape[0], f_h, f_w])

        n_fm, f_h, f_w, _, _ = infos[10]
        self.conv7_weights = op.initialize_weights([n_fm, self.conv6_weights.shape[0], f_h, f_w])

        n_fm, f_h, f_w, _, _ = infos[12]
        self.conv8_weights = op.initialize_weights([n_fm, self.conv7_weights.shape[0], f_h, f_w])

        n_fm, f_h, f_w, _, _ = infos[13]
        self.conv9_weights = op.initialize_weights([n_fm, self.conv8_weights.shape[0], f_h, f_w])

        n_fm, f_h, f_w, _, _ = infos[14]
        self.conv10_weights = op.initialize_weights([n_fm, self.conv9_weights.shape[0], f_h, f_w])

        self.gamma1, self.beta1 = op.initialize_weights([2])
        self.gamma2, self.beta2 = op.initialize_weights([2])
        self.gamma3, self.beta3 = op.initialize_weights([2])
        self.gamma4, self.beta4 = op.initialize_weights([2])
        self.gamma5, self.beta5 = op.initialize_weights([2])
        self.gamma6, self.beta6 = op.initialize_weights([2])
        self.gamma7, self.beta7 = op.initialize_weights([2])
        self.gamma8, self.beta8 = op.initialize_weights([2])
        self.gamma9, self.beta9 = op.initialize_weights([2])
