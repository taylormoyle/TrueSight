import tensorflow as tf
import numpy as np
import Operations_tf as op
import Neural_Network as nn
from PIL import Image
import glob
import os
import xml.etree.ElementTree as ET
import random as rand
import time as t
import random

RES = 208
DELTA_HUE = 0.032
MAX_DELTA_SATURATION = 1.5
MIN_DELTA_SATURATION = 0.5


# Gather files, attach labels to their associated filenames, and return an array of tuples [(filename, label)]
def prep_data(img_dir, xml_dir, test_percent=10, validation_percent=10):
    dataset = []
    labels = {}
    img_path = os.path.join(img_dir, "*")
    img_files = glob.glob(img_path)
    sorted(img_files)

    xml_path = os.path.join(xml_dir, "*")
    xml_files = glob.glob(xml_path)
    sorted(xml_files)

    for f, x in zip(img_files, xml_files):
        _, name = os.path.split(f)
        img_labels = []

        tree = ET.parse(x)
        root = tree.getroot()
        for elem in root:
            if elem.tag == 'object':
                label = []
                midpoints = []
                for subelem in elem:
                    if subelem.tag == 'name':
                        label.append(subelem.text)
                    if subelem.tag == 'bndbox':
                        xmin = 0
                        xmax = 0
                        ymin = 0
                        ymax = 0
                        for e in subelem:
                            if e.tag == 'xmax': xmax = int(float(e.text))
                            if e.tag == 'xmin': xmin = int(float(e.text))
                            if e.tag == 'ymax': ymax = int(float(e.text))
                            if e.tag == 'ymin': ymin = int(float(e.text))

                        midpoints.append((int(((ymax - ymin) / 2)), int(((xmax - xmin) / 2))))

                for l, m in zip(label, midpoints):
                    img_labels.append({'name': l, 'midpoint': m})

        #data = {'filename': f, 'labels': labels}
        dataset.append(f)
        labels[f] = img_labels

    num_validation_images = int(len(dataset) * (validation_percent / 100))
    num_test_images = int(len(dataset) * (test_percent / 100))

    validation_dataset = dataset[0:num_validation_images]
    test_dataset = dataset[num_validation_images:(num_test_images + num_validation_images)]
    training_dataset = dataset[(num_test_images + num_validation_images):]
    dataset = {'training': training_dataset, 'test': test_dataset, 'validation': validation_dataset}

    return dataset, labels


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
def load_img(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value, channels=3)
    image = resize_img(image)
    image = process_data(image)
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, key


def load_data(filenames, batch_size, num_epochs):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    image, label = load_img(filename_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
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

# Perform transformations, normalize images, return array of tuples [(norm_image, label)]
def process_data(images):
    # Perform Transformations on all images to diversify dataset
    image_huerized = tf.image.random_hue(images, DELTA_HUE)
    image_saturized = tf.image.random_saturation(image_huerized, MIN_DELTA_SATURATION, MAX_DELTA_SATURATION)
    image_flipperized = tf.image.random_flip_left_right(image_saturized)
    return image_flipperized


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
            p = rand.randint(0, len(w_re)-1)
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


infos = [[16, 3, 3, 1, 1],      # output shape (416, 416, 16)
         [16, 3, 3, 1, 1],      # output shape (416, 416, 16)
         [0, 2, 2, 2, 0],       # output shape (208, 208, 16)
         [32, 3, 3, 1, 1],      # output shape (208, 208, 32)
         [32, 3, 3, 1, 1],      # output shape (208, 208, 32)
         [0, 2, 2, 2, 0],       # output shape (104, 104, 32)
         [64, 3, 3, 1, 1],      # output shape (104, 104, 64)
         [0, 2, 2, 2, 0],       # output shape ( 52,  52, 64)
         [64, 3, 3, 1, 1],     # output shape ( 52,  52, 128)
         [0, 2, 2, 2, 0],       # output shape ( 26,  26, 128)
         [256, 3, 3, 1, 1],     # output shape ( 26,  26, 256)
         [0, 2, 2, 2, 0],       # output shape ( 13,  13, 256)
         [512, 1, 1, 1, 0],     # output shape ( 13,  13, 512)
         [512, 1, 1, 1, 0],     # output shape ( 13,  13, 512)
         [ 1, 1, 1, 1, 0]]      # output shape ( 13,  13, 5)


hypers = [7]
img_dir = "data\\VOC2012\\JPEGImages"
xml_dir = "data\\VOC2012\\Annotations"
save_dir = "models\\"
classes = ['person', 'dog', 'aeroplane', 'bus', 'bird', 'boat', 'car', 'bottle',
           'cat', 'horse', 'diningtable', 'cow', 'train', 'motorbike', 'bicycle',
           'sheep', 'tvmonitor', 'chair', 'sofa', 'pottedplant']

"""
net = nn.Neural_Network("facial_recognition", infos, hypers, training=True)
weights = net.init_facial_rec_weights(infos)
learning_rate = 0.001

data, lbls = prep_data(img_dir, xml_dir)
data['training'], labels = filter_data(lbls, classes)
for e in range(epoch):
    # shuffle data
    dataset = data['training']
    rand.shuffle(dataset)
    num_batches = int(len(dataset) / batch_size)
    for b in range(num_batches):
        # load batch
        imgs = load_data(dataset, batch_size, b)

        # process batch
        # implement later

        # forward prop
        predictions, cache = net.forward_prop(infos, imgs['images'], weights, training=True)

        # back prop
        grads = net.backward_prop(cache, predictions, imgs['labels'], weights, infos)

        if b % 5 == 0:
            rel_err = grad_check(net, imgs['images'], imgs['labels'], weights, grads, infos)
            for i in rel_err:
                print(e, b, i, 'gradient errors: ', rel_err[i])

        weights = net.update_weights(weights, grads, learning_rate, batch_size)
    # check validation accuracy

# check test accuracy

print(t.time() - s)

"""
'''    TENSORFLOW TRAINGING SCRIPT   '''

epochs = 10
batch_size = 96
learning_rate = 1e-5

dataset, labels = prep_data(img_dir, xml_dir)
train_data, train_labels = filter_data(dataset['training'], labels, classes)
val_data, val_labels = filter_data(dataset['validation'], labels, classes)
test_data, test_labels = filter_data(dataset['test'], labels, classes)

train_img_batch, train_label_batch = load_data(train_data, batch_size, epochs)
val_img_batch, val_label_batch = load_data(val_data, batch_size, None)
test_img_batch, test_label_batch = load_data(test_data, batch_size, None)

net = nn.Neural_Network("facial_recognition", infos, hypers)

inp_placeholder = tf.placeholder(tf.float32, shape=[None, 3, RES, RES])
training_placeholder = tf.placeholder(tf.bool)
pred_placeholder = net.forward_prop(infos, inp_placeholder, training=training_placeholder)

ground_truth_placeholder = tf.placeholder(tf.float32)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth_placeholder, logits=pred_placeholder))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(pred_placeholder, 1), tf.argmax(ground_truth_placeholder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

s = t.time()
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    num_train_batches = int(len(train_data) / batch_size)
    print(num_train_batches)
    for e in range(epochs):
        print("epoch: ", e)
        print('getting validation accuracy...')
        num_val_batches = int(len(val_data) / batch_size)
        train_accuracy = 0.
        for vb in range(num_val_batches):
            val_batch, val_lbl_keys = sess.run((val_img_batch, val_label_batch))
            val_lbls = [val_labels[l.decode()] for l in val_lbl_keys]
            train_accuracy += accuracy.eval(
                feed_dict={inp_placeholder: val_batch,
                           training_placeholder: False,
                           ground_truth_placeholder: val_lbls}) / float(num_val_batches)
            print("\r%d/%d. current training accuracy: %g" % (vb, num_val_batches, train_accuracy), end='')
        print("\ntraining accuracy: %g" % train_accuracy)
        for b in range(num_train_batches):
            train_batch, train_lbl_keys = sess.run((train_img_batch, train_label_batch))
            train_lbls = [train_labels[l.decode()] for l in train_lbl_keys]
            train_step.run(feed_dict={inp_placeholder: train_batch,
                                      training_placeholder: True,
                                      ground_truth_placeholder: train_lbls})

        if e % 25 == 0 and e != 0:
            # save checkpoint
            save_path = os.path.join(save_dir, "conv6_208_%g.ckpt" % train_accuracy)
            saver.save(sess, save_path)

    num_test_batches = int(len(test_data) / batch_size)
    test_accuracy = 0
    for _ in range(num_test_batches):
        test_batch, test_lbl_keys = sess.run((test_img_batch, test_label_batch))
        test_lbls = [test_labels[l.decode()] for l in test_lbl_keys]
        test_accuracy += accuracy.eval(
            feed_dict={inp_placeholder: test_batch,
                       training_placeholder: False,
                       ground_truth_placeholder: test_lbls})
    test_accuracy /= num_test_batches
    print("test accuracy: %g" % test_accuracy)

    # save end
    save_path = os.path.join(save_dir, "conv6_208_%g.ckpt" % test_accuracy)
    saver.save(sess, save_path)

    coord.request_stop()
    coord.join(threads)

print("runtime: ", (t.time()-s))
