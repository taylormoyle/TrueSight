import tensorflow as tf
import numpy as np
import Operations as op
import Neural_Network as nn
from PIL import Image
import glob
import os
import xml.etree.ElementTree as ET
import random as rand

RES = 416
DELTA_HUE = 0.3
MAX_DELTA_SATURATION = 0.3
MIN_DELTA_SATURATION = 0.3


# Gather files, attach labels to their associated filenames, and return an array of tuples [(filename, label)]
def prep_data(img_dir, xml_dir, test_percent=10, validation_percent=10):
    dataset = []
    img_path = os.path.join(img_dir, "*")
    img_files = glob.glob(img_path)
    sorted(img_files)

    xml_path = os.path.join(xml_dir, "*")
    xml_files = glob.glob(xml_path)
    sorted(xml_files)

    for f, x in zip(img_files, xml_files):
        _, name = os.path.split(f)
        labels = []
        # xml_file = name[:-4] + ".xml"

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
                    labels.append({'name': l, 'midpoint': m})

        data = {'filename': f, 'labels': labels}
        dataset.append(data)
        labels = []
        # print(data)

    rand.shuffle(dataset)

    num_validation_images = int(len(dataset) * (validation_percent / 100))
    num_test_images = int(len(dataset) * (test_percent / 100))

    validation_dataset = dataset[0:num_validation_images]
    test_dataset = dataset[num_validation_images:(num_test_images + num_validation_images)]
    training_dataset = dataset[(num_test_images + num_validation_images):]
    dataset = {'training': training_dataset, 'test': test_dataset, 'validation': validation_dataset}

    return dataset


# Loads each file as np RGB array, and returns an array of tuples [(image, label)]
def load_data(data, batch_size, batch):
    images = {'images': np.zeros((batch_size, 3, RES, RES)), 'labels': []}
    for i in range(batch_size):
        im = Image.open(data[i*batch]['filename'])
        img = resize_img(im)
        img = img.transpose((2, 0, 1))
        images['images'][i] = img
        images['labels'].append(data[i*batch]['labels'])
    return images


# Resize image to a 416 x 416 resolution
def resize_img(img):
    img.thumbnail([RES, RES], Image.ANTIALIAS)
    img = np.array(img)
    height, width, _ = img.shape
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    if height < RES:
        diff = RES - height
        pad = int(diff / 2)
        pad_top = pad
        if diff % 2 == 0:
            pad_bottom = pad
        else:
            pad_bottom = pad + 1
    if width < RES:
        diff = RES - width
        pad = int(diff / 2)
        pad_left = pad
        if diff % 2 == 0:
            pad_right = pad
        else:
            pad_right = pad + 1
    img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    return img


# Perform transformations, normalize images, return array of tuples [(norm_image, label)]
def process_data(images):
    # Resize all images to 416 x 416
    #for i in range(images.shape):
    #    resize_img(images[i])

    # Perform Transformations on all images to diversify dataset
    images = tf.transpose(images, (0, 2, 3, 1))
    image_huerized = tf.image.random_hue(images, DELTA_HUE)
    image_saturized = tf.image.random_saturation(image_huerized, MAX_DELTA_SATURATION, MIN_DELTA_SATURATION)
    image_flipperized = tf.image.random_flip_left_right(image_saturized)
    images = tf.transpose(image_flipperized, (0, 3, 1, 2))

    # Normalize images to reduce noise
    mean, variance = tf.nn.moments(images, axes=[0, 2, 3])
    images = (image_flipperized - mean) / tf.sqrt(variance)
    return images


def filter_data(dataset, classes):
    ret_data = []
    for img in dataset:
        if len(img['labels']) == 1:
            label = np.zeros(20)
            name = img['labels'][0]['name']
            l = classes.index(name)
            label[l] = 1

            ret_data.append({'filename': img['filename'], 'labels': label})
    return ret_data


def grad_check(net, inp, labels, weights, gradients, infos, epsilon=1e-5):
    rel_error = {}
    for w in weights:
        if w == 'full':
            continue
        back = weights[w].shape
        w_re = weights[w].reshape(-1)
        len_10 = int(len(w_re) * 1)

        n_g = np.zeros(len_10)
        a_g = np.zeros(len_10)
        for i in range(len_10):
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

            n_g[i] = (cost_plus - cost_minus) / (2 * epsilon)
            a_g[i] = gradients[w].reshape(-1)[p]

        num = np.abs(a_g - n_g)
        denom = np.abs(a_g) + np.abs(n_g)
        rel_error[w] = num / denom
    return rel_error

''''     TRAINING SCRIPT     '''
epoch = 100
batch_size = 2

infos = [[1, 3, 3, 1, 1],      # output shape (416, 416, 16)
         [1, 3, 3, 1, 1],      # output shape (416, 416, 16)
         [0, 2, 2, 2, 0],       # output shape (208, 208, 16)
         [1, 3, 3, 1, 1], #[32, 3, 3, 1, 1],      # output shape (208, 208, 32)
         [32, 3, 3, 1, 1],      # output shape (208, 208, 32)
         [0, 2, 2, 2, 0],       # output shape (104, 104, 32)
         [64, 3, 3, 1, 1],      # output shape (104, 104, 64)
         [0, 2, 2, 2, 0],       # output shape ( 52,  52, 64)
         [128, 3, 3, 1, 1],     # output shape ( 52,  52, 128)
         [0, 2, 2, 2, 0],       # output shape ( 26,  26, 128)
         [256, 3, 3, 1, 1],     # output shape ( 26,  26, 256)
         [0, 2, 2, 2, 0],       # output shape ( 13,  13, 256)
         [512, 1, 1, 1, 0],     # output shape ( 13,  13, 512)
         [512, 1, 1, 1, 0],     # output shape ( 13,  13, 512)
         [ 1, 1, 1, 1, 0]]      # output shape ( 13,  13, 5)


hypers = [7]
img_dir = "data\\VOC2012\\JPEGImages"
xml_dir = "data\\VOC2012\\Annotations"
classes = ['person', 'dog', 'aeroplane', 'bus', 'bird', 'boat', 'car', 'bottle',
           'cat', 'horse', 'diningtable', 'cow', 'train', 'motorbike', 'bicycle',
           'sheep', 'tvmonitor', 'chair', 'sofa', 'pottedplant']

net = nn.Neural_Network("facial_recognition", infos, hypers, training=True)
weights = net.init_facial_rec_weights(infos)

data = prep_data(img_dir, xml_dir)
data['training'] = filter_data(data['training'], classes)
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
        grads = net.backward_prop(cache, predictions, imgs['labels'], weights, infos, 0.001)

        if b % 5 == 0:
            rel_err = grad_check(net, imgs['images'], imgs['labels'], weights, grads, infos)
            for i in rel_err:
                print('gradient errors: ', rel_err[i])
    # check validation accuracy

# check test accuracy