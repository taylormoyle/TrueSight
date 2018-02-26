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
    labels = []
    img_path = os.path.join(img_dir, "*")
    img_files = glob.glob(img_path)
    sorted(img_files)

    xml_path = os.path.join(xml_dir, "*")
    xml_files = glob.glob(xml_path)
    sorted(xml_files)

    for f, x in zip(img_files, xml_files):
        _, name = os.path.split(f)
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
    images = []
    for i in range(batch_size):
        im = Image.open(data[i*batch]['filename'])
        img = np.array(im)
        img = resize_img(img)
        img = img.transpose((2, 0, 1))
        images.append({'img': img, 'label': data[i*batch]['label']})
    return images


# Resize image to a 416 x 416 resolution
def resize_img(img):
    RES = 416
    img.thumbnail([RES, RES], Image.ANTIALIAS)
    img = np.array(img)
    height, width, _ = img.shape
    height_pad = 0
    width_pad = 0
    if height < RES:
        height_pad = int((RES - height) / 2)
    if width < RES:
        width_pad = int((RES - width) / 2)
    img = np.pad(img, ((height_pad, height_pad), (width_pad, width_pad), (0, 0)), mode='constant')
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


def grad_check(net, inp, labels, weights, gradients, params, infos, epsilon=1e-8):
    batch, lcoord, lnoobj = params
    #back = weights.shape
    #W = weights.reshape(-1)
    #num_grads = {}
    #an_grads = {}
    rel_error = {}

    for w in weights:
        back = weights[w].shape
        w_re = weights[w].reshape(-1)
        len_10 = int(len(w_re) * 0.1)

        n_g = np.zeros(len_10)
        a_g = np.zeros(len_10)
        for i in range(len_10):
            p = rand.randint(0, len(w_re)-1)

            w_re[p] += epsilon
            w_re = w_re.reshape(back)
            weights[w] = w_re

            cost_plus = net.forward_prop(infos, inp, weights, training=False)
            cost_plus = op.mean_square_error(cost_plus, labels)

            w_re = w_re.reshape(-1)
            w_re[p] -= 2 * epsilon
            w_re = w_re.reshape(back)
            weights[w] = w_re

            cost_minus = net.forward_prop(infos, inp, weights, training=False)
            cost_minus = op.mean_square_error(cost_minus, labels)

            w_re= w_re.reshape(-1)

            n_g[i] = (cost_plus - cost_minus) / (2 * epsilon)
            a_g[i] = gradients[w].reshape(-1)[p]
        #num_grads[w] = n_g
        #an_grads[w] = a_g

        rel_error[w] = np.linalg.norm(a_g - n_g) / (np.linalg.norm(a_g) + np.linalg.norm(n_g))
    return rel_error


