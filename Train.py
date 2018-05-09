import tensorflow as tf
import numpy as np
import Operations_tf as op
import Neural_Network as nn
import os
import time as t
import random

from glob import glob
from tensorflow.python import debug as tf_debug

RES = 208
DELTA_HUE = 0.32
MAX_DELTA_SATURATION = 1.25
MIN_DELTA_SATURATION = 0.75
MAX_DELTA_BRIGHTNESS = 0.3
MAX_CONTRAST = 1.5
MIN_CONTRAST = 0.5


def prep_classification_data(data_dir, focus):
    '''
    Prepares dataset by matching filenames with their corresponding labels
    :param data_dir: The directory of the dataset
    :return: Data: Dictionary of filenames, separated by test, train, validaiton.
             Labels: Dictionary of labels with filenames as key.
    '''
    data = {'train': {}, 'test': {}, 'validation': {}}
    labels = {}
    class_path = os.path.join(data_dir, '*')
    classes = glob(class_path)
    for c in classes:
        files = []
        _ , obj = os.path.split(c)
        search_path = os.path.join(c, '*')
        for f in glob(search_path):
            files.append(f)
            label = [1] if obj == focus else [0]
            labels[f] = label

        random.shuffle(files)

        which = 'pos' if obj == focus else 'neg'

        train_size = int(len(files) * 0.8)
        val_size = int(len(files) * 0.1)
        data['train'][which] = files[:train_size]
        data['validation'][which] = files[train_size:train_size+val_size]
        data['test'][which] = files[train_size+val_size:]

    return data, labels


def load_img(filename_queue):
    '''
    Loads jpeg image and handsoff to process for data augmentation
    :param filename_queue: TensorFlow queue
    :return: Decoded image and filename
    '''
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value, channels=3)
    image.set_shape([RES, RES, 3])
    image = process_data(image)
    #image = tf.transpose(image, perm=[2, 0, 1])
    return image, key


def load_data(filenames, batch_size, perc_pos, training=True):
    '''
    Constructs TensorFlow filename queue and batch-shuffler thread-runners
    :param filenames:
    :param batch_size:
    :param num_epochs:
    :param training:
    :return: Image batch and Label batch thread-runners
    '''

    # create positive queue
    pos_filenames = filenames['pos']
    pos_queue = tf.train.string_input_producer(pos_filenames, shuffle=False)
    pos_image, pos_label = load_img(pos_queue)

    # create neg queue
    neg_filenames = filenames['neg']
    neg_queue = tf.train.string_input_producer(neg_filenames, shuffle=False)
    neg_image, neg_label = load_img(neg_queue)

    # collect batches
    pos_batch_size = int(batch_size * perc_pos)
    neg_batch_size = batch_size - pos_batch_size

    # get pos batch
    pos_image_batch, pos_label_batch = tf.train.batch([pos_image, pos_label],
                                                      batch_size=pos_batch_size,
                                                      capacity=pos_batch_size
    )

    # get neg batch
    neg_image_batch, neg_label_batch = tf.train.batch([neg_image, neg_label],
                                                      batch_size=neg_batch_size,
                                                      capacity=neg_batch_size
    )

    # join pos and neg batches
    image_batch = tf.concat([pos_image_batch, neg_image_batch], axis=0)
    label_batch = tf.concat([pos_label_batch, neg_label_batch], axis=0)

    return image_batch, label_batch


# Unimplemented
def shift_image(image):
    shift_x = tf.random_uniform([1], minval=-10, maxval=10, dtype=tf.int32)
    shift_y = tf.random_uniform([1], minval=-10, maxval=10, dtype=tf.int32)

    x_offest, x_pad = tf.cond(tf.greater_equal(shift_x, 0),
                              lambda: ([0, RES-shift_x], [shift_x, 0]),
                              lambda: ([-shift_x, RES], [0, -shift_x]))
    y_offset, y_pad = tf.cond(tf.greater_equal(shift_y, 0),
                              lambda: ([0, RES-shift_y], [shift_y, 0]),
                              lambda: ([-shift_y, RES], [0, -shift_y]))
    shifted_image = image[y_offset[0]:y_offset[1], x_offest[0]:x_offest[1], :]
    padded_shifted_image = tf.pad(shifted_image, [y_pad, x_pad], "CONSTANT")
    return padded_shifted_image


def process_data(image):
    '''
    Performs transformations to pseudo-expand dataset
    :param images:
    :return: Augmented Images
    '''

    norm_img = tf.image.per_image_standardization(image)

    # Perform Transformations on all images to diversify dataset
    #image_huerized = tf.image.random_hue(images, DELTA_HUE)
    #image_saturized = tf.image.random_saturation(image_huerized, MIN_DELTA_SATURATION, MAX_DELTA_SATURATION)
    #image_contrasterized = tf.image.random_contrast(image_saturized, MIN_CONTRAST, MAX_CONTRAST)
    image_flipperized = tf.image.random_flip_left_right(norm_img)

    brightness_min = 1.0 - (MAX_DELTA_BRIGHTNESS / 100)
    brightness_max = 1.0 - (MAX_DELTA_BRIGHTNESS / 100)
    bright_value = tf.random_uniform([1], minval=brightness_min, maxval=brightness_max)
    image_brighterized = tf.multiply(tf.cast(image_flipperized, dtype=tf.float32), bright_value)
    return image_brighterized


def grad_check(net, inp, labels, weights, gradients, infos, epsilon=1e-5):
    '''
    Calculates numerical gradients and compares against analytical grtadients by calculating relative error
    :param net: Neural Net Graph
    :param inp: Image
    :param labels: Labels
    :param weights: Learnable Paramaters
    :param gradients: Analytical Gradients calculated by back-prop
    :param infos: Arthitecture of Network
    :param epsilon: Small floating point value, used for numerical stability
    :return: Relative Error
    '''
    rel_error = {}
    check_num_grads = 5
    for w in weights:
        print("weights %s.." % w)
        back = weights[w].shape
        w_re = weights[w].reshape(-1)

        n_g = np.zeros(check_num_grads)
        a_g = np.zeros(check_num_grads)
        for i in range(check_num_grads):
            p = random.randint(0, len(w_re)-1)
            w_re[p] += epsilon
            w_re = w_re.reshape(back)
            weights[w] = w_re

            cost_plus = net.forward_prop(infos, inp, weights, training=False)
            cost_plus = op.mean_square_error(cost_plus, labels)
            cost_plus = np.sum(cost_plus) / inp.shape[0]

            w_re = w_re.reshape(-1)
            w_re[p] -= 2 * epsilon
            w_re = w_re.reshape(back)
            weights[w] = w_re

            cost_minus = net.forward_prop(infos, inp, weights, training=False)
            cost_minus = op.mean_square_error(cost_minus, labels)
            cost_minus = np.sum(cost_minus) / inp.shape[0]

            w_re= w_re.reshape(-1)
            w_re[p] += epsilon

            n_g[i] = (cost_plus - cost_minus) / (2 * epsilon)
            a_g[i] = gradients[w].reshape(-1)[p]

        num = np.linalg.norm(a_g - n_g)
        denom = np.linalg.norm(a_g) + np.linalg.norm(n_g)
        rel_error[w] = num / denom
    return rel_error

'''     TRAINING SCRIPT     '''

architecture = {'conv1': [16, 3, 3, 3, 1, 'SAME'], 'pool1': [2, 2, 2, 0],       # output shape 150
                'conv2': [32, 16, 3, 3, 1, 'VALID'], 'pool2': [2, 2, 2, 0],     # output shape 74
                'conv3': [64, 32, 3, 3, 1, 'VALID'], 'pool3': [2, 2, 2, 0],     # output shape 36
                'conv4': [128, 64, 3, 3, 1, 'VALID'], 'pool4': [2, 2, 2, 0],    # output shape 17
                'conv5': [256, 128, 3, 3, 1, 'VALID'],                          # output shape 15
                'conv6': [512, 256, 1, 1, 1, 'VALID'],                          # output shape 15
                'conv7': [1024, 512, 1, 1, 1, 'VALID'],                         # output shape 15
                'conv8': [5, 1024, 1, 1, 1, 'VALID']                            # output shape 15
                }

# Specify where to load data and to save models and logs
data_dir = "data\\classification\\classification"
save_dir = "models"
log_dir = "logs"
conf = 0.50
neg_size = 33000    # amount of negative examples

'''    TENSORFLOW TRAINGING SCRIPT   '''

# Training Hyperparamaters

epochs = 5000
batch_size = 32
pos_batch_perc = 0.25
initial_learning_rate = 0.001
steps = [20, 40, 60, 80, 400, 1000]
scales = [1.5, 1.5, 2, 3, 0.5, 0.1]
ending_learning_rate = 1e-5
decay_steps = 7500
power = 4
momentum = 0.9
weight_decay = 0.005
pos_weight = 1.5
epsilon = 1e-8
test_iters = 100
val_batch_size = 50
test_batch_size = 50


# Prepare data
dataset, labels = prep_classification_data(data_dir, 'face')

# Create them training, test and validaiton load-runners
with tf.device('/cpu:0'):
    test_img_batch, test_label_batch = load_data(dataset['test'], test_batch_size, pos_batch_perc, training=False)
    val_img_batch, val_label_batch = load_data(dataset['validation'], val_batch_size, pos_batch_perc, training=False)
train_img_batch, train_label_batch = load_data(dataset['train'], batch_size, pos_batch_perc)

# Constructing the face_rec graph
inp_placeholder = tf.placeholder(tf.float32, shape=[None, RES, RES, 3])
training_placeholder = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)
fac_rec_preds, layer = nn.create_test_rec(inp_placeholder, architecture, keep_prob, training=training_placeholder)

# Calculate loss using cross entropy
ground_truth_placeholder = tf.placeholder(tf.int32)
with tf.name_scope('loss'):
    loss = tf.nn.weighted_cross_entropy_with_logits(tf.cast(ground_truth_placeholder, dtype=tf.float32),
                                                    fac_rec_preds, pos_weight=pos_weight)
    l2_reg = tf.reduce_sum([tf.nn.l2_loss(w) for w in tf.trainable_variables()
                            if 'bias' not in w.name])
    l2_loss = tf.reduce_mean(loss) + weight_decay * l2_reg
#tf.summary.scalar('loss', loss)


with tf.name_scope('training'):

    '''
    # Decay the learning rate
    learning_rate = tf.convert_to_tensor(initial_learning_rate, dtype=tf.float32)
    global_step = tf.placeholder(tf.float32)
    decay_steps = tf.convert_to_tensor(decay_steps, dtype=tf.float32)
    ending_learning_rate = tf.convert_to_tensor(ending_learning_rate, dtype=tf.float32)
    power = tf.convert_to_tensor(power, dtype=tf.float32)
    global_decay = tf.minimum(global_step, decay_steps)

    decayed_rate = (learning_rate - ending_learning_rate) * \
                   tf.pow((1 - global_decay / decay_steps), power) + \
                   ending_learning_rate

    # Get optimizer and gradients
    optimizer = tf.train.MomentumOptimizer(decayed_rate, momentum=momentum)
    gradients, variables = zip(*optimizer.compute_gradients(l2_loss))

    # Clip gradients to prevent exploding
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    gradients = [tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad) if grad is not None
                 else grad for grad in gradients]

    # Apply gradients
    train_step = optimizer.apply_gradients(zip(gradients, variables))
    '''
    learning_rate = tf.placeholder(tf.float32)
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(l2_loss)


with tf.name_scope('accuracy'):
    # Calculate Accuracy
    logits = tf.where(fac_rec_preds > conf, tf.ones_like(fac_rec_preds), tf.zeros_like(fac_rec_preds))
    correct_prediction = tf.equal(tf.round(fac_rec_preds),
                                  tf.cast(ground_truth_placeholder, dtype=tf.float32))
    num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

# Creates saver and merged summaries
saver = tf.train.Saver()
merged = tf.summary.merge_all()

s = t.time()


with tf.Session() as sess:
    # Wrap session in debugger
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # Create summary file-writers
    test_log_dir = os.path.join(log_dir, 'test')
    train_log_dir = os.path.join(log_dir, 'train')
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    #test_writer = tf.summary.FileWriter(test_log_dir, sess.graph)

    # Initialize global and local variables and start thread-runners
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #nn.load_model(sess, 'models\\conv6_416_0.9286.ckpt')


    for e in range(epochs):
        if e % 200 == 0:
            # Run test against validation dataset
            val_correct = 0.
            val_error = 0.
            print('\n\033[93mgetting validation accuracy...')
            for _ in range(test_iters):
                val_batch, val_lbl_keys = sess.run((val_img_batch, val_label_batch))
                val_lbls = [labels[l.decode()] for l in val_lbl_keys]
                correct, error= sess.run([num_correct, l2_loss], feed_dict={inp_placeholder: val_batch,
                                                                            training_placeholder: False,
                                                                            keep_prob: 1.0,
                                                                            ground_truth_placeholder: val_lbls})
                val_correct += correct
                val_error += error
            print("\033[93m%d/%d correct on validation\terror: %f" % (val_correct,
                                                                      val_batch_size*test_iters,
                                                                      tf.reduce_mean(val_error).eval()))
            #test_writer.add_summary(summary, global_step=(e*num_val_batches + vb))
            print("\033[93mtraining accuracy: %g" % (val_correct / (val_batch_size*test_iters)))

            if e % 500 == 0 and e != 0:
                # Save checkpoint
                print("\033[93mSaving checkpoint...")
                save_path = os.path.join(save_dir, "conv6_416_%g.ckpt" % val_correct)
                saver.save(sess, save_path)

        # Run training step
        for i,j in zip(steps, scales):
            if e == i:
                initial_learning_rate *= j
        train_batch, train_lbl_keys = sess.run((train_img_batch, train_label_batch))
        train_lbls = [labels[l.decode()] for l in train_lbl_keys]
        summary, maps, _ = sess.run([merged, layer, train_step], feed_dict={inp_placeholder: train_batch,
                                                               training_placeholder: True,
                                                               keep_prob: 0.5,
                                                               ground_truth_placeholder: train_lbls,
                                                               learning_rate: initial_learning_rate})
        print("\r\033[0m%d/%d training steps.." % (e+1, epochs), end='')
        #train_writer.add_summary(summary, global_step=e)

    # Run against test dataset
    print("\n\033[94mrunning test accuracy..")
    test_correct = 0
    for _ in range(test_iters):
        test_batch, test_lbl_keys = sess.run((test_img_batch, test_label_batch))
        test_lbls = [labels[l.decode()] for l in test_lbl_keys]
        test_correct += num_correct.eval(
            feed_dict={inp_placeholder: test_batch,
                       training_placeholder: False,
                       keep_prob: 1.0,
                       ground_truth_placeholder: test_lbls})
    test_accuracy = test_correct / float(test_iters*test_batch_size)
    print("\033[94mtest accuracy: %g" % test_accuracy)

    # Save end
    print("\033[93mSaving final train run..")
    save_path = os.path.join(save_dir, "conv6_416_%g.ckpt" % test_accuracy)
    saver.save(sess, save_path)

    # join test threads
    coord.request_stop()
    coord.join(threads)

print("runtime: ", (t.time()-s))
