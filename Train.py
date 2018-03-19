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
DELTA_HUE = 0.2
MAX_DELTA_SATURATION = 1.5
MIN_DELTA_SATURATION = 0.5
MAX_DELTA_BRIGHTNESS = 0.2


def prep_classification_data(data_dir):
    data = {'train': [], 'test': [], 'validation': []}
    labels = {'train': {}, 'test': {}, 'validation': {}}
    for d in data:
        class_path = os.path.join(data_dir, d, '*')
        classes = glob(class_path)
        for c,l in zip(classes, range(len(classes))):
            _ , obj = os.path.split(c)
            search_path = os.path.join(c, '*')
            for f in glob(search_path):
                if '.jpg' in f:
                    data[d].append(f)
                    label = np.array([1, 0]) if obj == 'faces' else np.array([0, 1])
                    labels[d][f] = label
        random.shuffle(data[d])
    return data, labels

def filter_data(filenames, labels, classes):
    ret_imgs = []
    ret_lbls = {}
    for img in filenames:
        if len(labels[img]) == 1:
            label = np.zeros(len(classes))
            name = labels[img][0]['name']
            lbl = classes.index(name)
            label[lbl] = 1

            ret_imgs.append(img)
            ret_lbls[img] = label
    return ret_imgs, ret_lbls

def prep_face_data(train_fn, validation_fn, data_directory):
    dataset = {'training': [], 'test': [], 'validation': []}
    train_labels = {}
    valtest_labels = {}

    with open(train_fn) as f:
        filename = ""
        count = 0
        line = f.readline()
        while not line == "":
            if ".jpg" in line:
                name = line.split('/')
                filename = os.path.join(data_directory, 'WIDER_train', 'images', name[0], name[1])
                dataset['training'].append(filename[:-1])
                count = f.readline()
            label = []
            for b in range(int(count)):
                box = f.readline().split()
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])
                label.append([1, x, y, w, h])
            train_labels[filename[:-1]] = label
            line = f.readline()

    with open(validation_fn) as f:
        filename = ""
        count = 0
        line = f.readline()
        while not line == "":
            if ".jpg" in line:
                name = line.split('/')
                filename = os.path.join(data_directory, 'WIDER_val', 'images', name[0], name[1])
                dataset['validation'].append(filename[:-1])
                count = f.readline()
            label = []
            for b in range(int(count)):
                box = f.readline().split()
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])
                label.append([1, x, y, w, h])
            valtest_labels[filename[:-1]] = label
            line = f.readline()

    random.shuffle(dataset['validation'])
    dataset['test'] = dataset['validation'][:int(len(dataset['validation']) / 2)]
    dataset['validation'] = dataset['validation'][int(len(dataset['validation']) / 2):]
    return dataset, train_labels, valtest_labels


# Loads each file as np RGB array, and returns an array of tuples [(image, label)]
def load_img(filename_queue, training):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value, channels=3)
    image.set_shape([RES, RES, 3])
    #image = resize_img(image)
    image = process_data(image)
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, key


def load_data(filenames, batch_size, num_epochs, training=True):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    image, label = load_img(filename_queue, training)
    min_after_dequeue = 1000 if training else 49
    capacity = (min_after_dequeue + 3 * batch_size) if training else batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return image_batch, label_batch


# Resize image to a 416 x 416 resolution
def resize_img(img):
    h, w, _ = img.get_shape()
    new_h = 0
    new_w = 0
    if h > w:
        new_h = RES
        new_w = int((w / h) * new_h)
    elif w > h:
        new_w = RES
        new_h = int((w / h) / new_w)
    else:
        new_h = RES
        new_w = RES

    img_resized = tf.image.resize_images(img, [new_h, new_w])
    img_resized = tf.image.resize_image_with_crop_or_pad(img_resized, RES, RES)

    return img_resized

def shift_image(image):
    shift_x = tf.random_uniform([1], minval=-10, maxval=10, dtype=tf.int32)
    shift_y = tf.random_uniform([1], minval=-10, maxval=10, dtype=tf.int32)

# Perform transformations, normalize images, return array of tuples [(norm_image, label)]
def process_data(images):
    # Perform Transformations on all images to diversify dataset
    image_brighterized = tf.image.random_brightness(images, MAX_DELTA_BRIGHTNESS)
    image_huerized = tf.image.random_hue(image_brighterized, DELTA_HUE)
    image_saturized = tf.image.random_saturation(image_huerized, MIN_DELTA_SATURATION, MAX_DELTA_SATURATION)
    image_flipperized = tf.image.random_flip_left_right(image_saturized)
    return image_flipperized


def grad_check(net, inp, labels, weights, gradients, infos, epsilon=1e-5):
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

''''     TRAINING SCRIPT     '''

architecture = {'conv1': [16, 3, 3, 3, 1, 1], 'pool1': [2, 2, 2, 0],    # output shape 104
                'conv2': [32, 16, 3, 3, 1, 1], 'pool2': [2, 2, 2, 0],    # output shape 52
                'conv3': [64, 32, 3, 3, 1, 1], 'pool3': [2, 2, 2, 0],    # output shape 26
                'conv4': [128, 64, 3, 3, 1, 1], 'pool4': [2, 2, 2, 0],   # output shape 13
                'conv5': [256, 128, 3, 3, 1, 1],
                'full':  [13*13*256, 2]
                }


hypers = [7]
data_dir = "data\\classification"
save_dir = "models"
log_dir = "logs"

'''    TENSORFLOW TRAINGING SCRIPT   '''

epochs = 200
batch_size = 96
initital_learning_rate = 0.001
ending_learning_rate = 1e-5
decay_steps = 100
power = 4
momentum = 0.0
weight_decay = 0.5
epsilon = 1e-8
val_batch_size = 50
test_batch_size = 50

dataset, labels = prep_classification_data(data_dir)

with tf.device('/cpu:0'):
    val_img_batch, val_label_batch = load_data(dataset['validation'], val_batch_size, None, training=False)
    test_img_batch, test_label_batch = load_data(dataset['test'], test_batch_size, None, training=False)
train_img_batch, train_label_batch = load_data(dataset['train'], batch_size, epochs)

inp_placeholder = tf.placeholder(tf.float32, shape=[None, 3, RES, RES])
training_placeholder = tf.placeholder(tf.bool)
fac_rec_preds = nn.create_facial_rec(inp_placeholder, architecture, training=training_placeholder)

ground_truth_placeholder = tf.placeholder(tf.int32)
with tf.name_scope('loss'):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth_placeholder, logits=fac_rec_preds))
tf.summary.scalar('loss', cross_entropy_mean)

with tf.name_scope('training'):

    # decay learning rate..
    learning_rate = tf.convert_to_tensor(initital_learning_rate, dtype=tf.float32)
    global_step = tf.placeholder(tf.float32)
    decay_steps = tf.convert_to_tensor(decay_steps, dtype=tf.float32)
    ending_learning_rate = tf.convert_to_tensor(ending_learning_rate, dtype=tf.float32)
    power = tf.convert_to_tensor(power, dtype=tf.float32)
    global_decay = tf.minimum(global_step, decay_steps)

    decayed_rate = (learning_rate - ending_learning_rate) * \
                   tf.pow((1 - global_decay / decay_steps), power) + \
                   ending_learning_rate

    # get optimizer and gradients
    optimizer = tf.train.MomentumOptimizer(decayed_rate, momentum=momentum)
    
    gradients, variables = zip(*optimizer.compute_gradients(cross_entropy_mean))

    # clip gradients to prevent from exploding
    #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    gradients = [tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad) if grad is not None
                 else grad for grad in gradients]

    train_step = optimizer.apply_gradients(zip(gradients, variables))

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(fac_rec_preds, 1), tf.argmax(ground_truth_placeholder, 1))
    num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

merged = tf.summary.merge_all()

s = t.time()
with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    test_log_dir = os.path.join(log_dir, 'test')
    train_log_dir = os.path.join(log_dir, 'train')
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    #test_writer = tf.summary.FileWriter(test_log_dir, sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for e in range(epochs):
        if e % 100 == 0:
            # run test against validation dataset
            val_correct = 0.
            print('\n\033[93mgetting validation accuracy...')
            for _ in range(int(4000/test_batch_size)):
                val_batch, val_lbl_keys = sess.run((val_img_batch, val_label_batch))
                val_lbls = [labels['validation'][l.decode()] for l in val_lbl_keys]
                val_correct += sess.run(num_correct, feed_dict={inp_placeholder: val_batch,
                                                               training_placeholder: False,
                                                               ground_truth_placeholder: val_lbls})
            print("\033[93m%d/%d correct on validation" % (val_correct, val_batch_size*int(4000/test_batch_size)))
            #test_writer.add_summary(summary, global_step=(e*num_val_batches + vb))
            print("\033[93mtraining accuracy: %g" % (val_correct / (val_batch_size*int(4000/test_batch_size))))

            if e % 500 == 0 and e != 0:
                # save checkpoint
                print("\033[93mSaving checkpoint...")
                save_path = os.path.join(save_dir, "conv6_208_%g.ckpt" % val_correct)
                saver.save(sess, save_path)

        # run training step
        train_batch, train_lbl_keys = sess.run((train_img_batch, train_label_batch))
        train_lbls = [labels['train'][l.decode()] for l in train_lbl_keys]
        summary, _ = sess.run([merged, train_step], feed_dict={inp_placeholder: train_batch,
                                                               training_placeholder: True,
                                                               ground_truth_placeholder: train_lbls,
                                                               global_step: e})
        print("\r\033[0m%d/%d training steps.." % (e+1, epochs), end='')
        train_writer.add_summary(summary, global_step=e)

    # run against test dataset
    print("\n\033[94mrunning test accuracy..")
    num_test_batches = int(len(dataset['test']) / test_batch_size)
    test_correct = 0
    for _ in range(num_test_batches):
        test_batch, test_lbl_keys = sess.run((test_img_batch, test_label_batch))
        test_lbls = [labels['test'][l.decode()] for l in test_lbl_keys]
        test_correct += num_correct.eval(
            feed_dict={inp_placeholder: test_batch,
                       training_placeholder: False,
                       ground_truth_placeholder: test_lbls})
    test_accuracy = test_correct / float(num_test_batches*test_batch_size)
    print("\033[94mtest accuracy: %g" % test_accuracy)

    # save end
    print("\033[93mSaving final train run..")
    save_path = os.path.join(save_dir, "conv6_208_%g.ckpt" % test_accuracy)
    saver.save(sess, save_path)

    coord.request_stop()
    coord.join(threads)

print("runtime: ", (t.time()-s))
