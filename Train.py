import tensorflow as tf
import numpy as np
from math import sqrt
import Operations as op
import Neural_Network as nn
from PIL import Image
import glob
import os
import xml.etree.ElementTree as ET

RES = 416
DELTA_HUE = 0.3
MAX_DELTA_SATURATION = 0.3
MIN_DELTA_SATURATION = 0.3

# Gather files, attach labels to their associated filenames, and return an array of tuples [(filename, label)]
def prep_data(img_dir, xml_dir):
    dataset = []
    labels = []
    img_path = os.path.join(img_dir, "*")
    img_files = glob.glob(img_path)
    # Loop through images and parse the name of the file, then pass it to the next loop for comparison.
    for f in img_files:
        _, name = os.path.split(f)
        xml_file = name[:-4] + ".xml"
        # Now we have the filename for the xml. Next up is searching for that xml file, traversing through it, finding the object label, appending that to label, then storing that in the tuple with its corresponding filename in a dictionary.
        os.chdir(xml_dir)
        for file in glob.glob('*.xml'):
            if file == xml_file:
                tree = ET.parse(file)
                root = tree.getroot()
                for elem in root:
                    if elem.tag == 'object':
                        for subelem in elem:
                            if subelem.tag == 'name':
                                labels.append(subelem.text)
        data = {'filename': f, 'labels': labels}
        dataset.append(data)
        labels = []
    return dataset


# Loads each file as np RGB array, and returns an array of tuples [(image, label)]
def load_data(data, batch_size, batch):
    images = []
    for i in range(batch_size):
        im = Image.open(data[i*batch]['filename'])
        img = np.array(im)
        img = img.transpose(2, 0, 1)
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
    for i in range(images.shape):
        resize_img(images[i])

    # Perform Transformations on all images to diversify dataset
    tf.transpose(0, 2, 3, 1)
    image_huerized = tf.image.random_hue(images, DELTA_HUE)
    image_saturized = tf.image.random_saturation(image_huerized, MAX_DELTA_SATURATION, MIN_DELTA_SATURAITON)
    image_flipperized = tf.image.random_flip_left_right(image_saturized)

    # Normalize images to reduce noise
    mean, variance = tf.nn.moments(images, axes=[0, 2, 3])
    images = (images - mean) / tf.sqrt(variance)
    return images


def grad_check(inp, labels, weights, grads, params, epsilon=1e-8):
    batch, lcoord, lnoobj = params
    back = weights.shape
    W = weights.reshape(-1)
    num_grads = tf.zeros(W.shape, dtype=np.float32)

    for p in range(len(W)):
        W[p] += epsilon
        W = W.reshape(back)
        cost_plus = nn.forward_prop(inp)
        cost_plus = op.cost_function(cost_plus, labels, batch, lcoord, lnoobj)
        W = W.reshape(-1)

        W[p] -= 2 * epsilon
        W = W.reshape(back)
        cost_minus = nn.forward_prop(inp)
        cost_minus = op.cost_function(cost_minus, labels, batch, lcoord, lnoobj)
        W = W.reshape(-1)

        num_grads[p] = (cost_plus - cost_minus) / (2 * epsilon)

    rel_error = np.linalg.norm(grads - num_grads) / (np.linalg.norm(grads) + np.linalg.norm(num_grads))
    return rel_error
