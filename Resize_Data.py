import random
from math import floor
from PIL import Image
import os
import numpy as np
import random
import glob
import xml.etree.ElementTree as ET

IMG_HEIGHT = 416
IMG_WIDTH = 416
CELL_HEIGHT = 16
CELL_WIDTH = 16
GRID_HEIGHT = 13
GRID_WIDTH = 13
RES = 416


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
    train_dataset = dataset[(num_test_images + num_validation_images):]
    dataset = {'train': train_dataset, 'test': test_dataset, 'validation': validation_dataset}

    return dataset, labels


def prep_face_data(train_fn, validation_fn, data_directory):
    dataset = {'train': [], 'test': [], 'validation': []}
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
                dataset['train'].append(filename[:-1])
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


def prep_all_face_data(img_dir, xml_dir, test_percent=10, validation_percent=10):
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
                for subelem in elem:
                    if subelem.tag == 'name':
                        img_labels.append(subelem.text)

        #data = {'filename': f, 'labels': labels}
        if 'person' not in img_labels:
            dataset.append(f)
            labels[f] = img_labels

    num_validation_images = int(len(dataset) * (validation_percent / 100))
    num_test_images = int(len(dataset) * (test_percent / 100))

    validation_dataset = dataset[0:num_validation_images]
    test_dataset = dataset[num_validation_images:(num_test_images + num_validation_images)]
    train_dataset = dataset[(num_test_images + num_validation_images):]
    dataset = {'train': train_dataset, 'test': test_dataset, 'validation': validation_dataset}

    return dataset, labels


def prep_temp_noface_data(img_dir, test_percent=10, validation_percent=10):
    dataset = []
    labels = {}
    img_path = os.path.join(img_dir, "*")
    img_files = glob.glob(img_path)
    sorted(img_files)

    for f in img_files:
        _, name = os.path.split(f)
        dataset.append(f)

    num_validation_images = int(len(dataset) * (validation_percent / 100))
    num_test_images = int(len(dataset) * (test_percent / 100))

    validation_dataset = dataset[0:num_validation_images]
    test_dataset = dataset[num_validation_images:(num_test_images + num_validation_images)]
    train_dataset = dataset[(num_test_images + num_validation_images):]
    dataset = {'train': train_dataset, 'test': test_dataset, 'validation': validation_dataset}

    return dataset, labels


def prep_classification_data(data_dir):
    datasets = {'train': [], 'test': [], 'validation': []}
    path = os.path.join(data_dir, '*', '*.jpg')
    images = glob.glob(path)
    random.shuffle(images)

    datasets['train'] = images[:800]
    datasets['test'] = images[800:900]
    datasets['validation'] = images[900:]
    return datasets

def prep_chockpnt_data(data_dir):
    dataset = {'face':[], 'noface':[]}
    search_files = os.path.join(data_dir, 'groundtruth', "P1*")
    label_files = glob.glob(search_files)

    for f in label_files:
        tree = ET.parse(f)
        root = tree.getroot()
        img_folder = root.attrib['name']
        for elem in root:
            filename = elem.get('number')
            img_file = os.path.join(data_dir, img_folder, filename + '.jpg')
            if len(elem) > 0:
                dataset['face'].append(img_file)
            else:
                dataset['noface'].append(img_file)
    return dataset


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
    return img, pad_top, pad_left


def process_data(data, labels, data_set, save_dir):
    j = 1
    #set_annotations_file = os.path.join(save_dir, "annotations_%s.txt" % data_set)
    #with open(set_annotations_file, 'w') as f:
    num_imgs = len(data)
    for i in data:
        #f.write(i + "\n")
        # load images, resize, and save resized images in save_dir
        im = Image.open(i)
        og_w, og_h = im.size
        img, pad_top, pad_left = resize_img(im)
        image = Image.fromarray(img)

        img_folder, filename = os.path.split(i)
        _, img_folder = os.path.split(img_folder)

        #img_save_dir = os.path.join(save_dir, img_folder)
        #if not os.path.exists(img_save_dir):
        #    os.mkdir(img_save_dir)

        file_pathname = os.path.join(save_dir, img_folder + filename)
        image.save(file_pathname)

        print("\r%d/%d processed.." % (j, num_imgs), end='')
        j += 1

        '''
        # get scaled labels and save new labels to new annotation file
        img_labels = labels[i]
        num_labels = len(img_labels)
        f.write(str(num_labels) + "\n")

        for lbl in img_labels:
            _, x, y, w, h = lbl

            new_x, new_y, new_h, new_w = rescale_labels(x, y, w, h, og_w, og_h, pad_top, pad_left)
            cell, x_mid, y_mid, bb_w, bb_h = find_midpoint([new_x, new_y, new_w, new_h])

            line = "%d, %f, %f, %f, %f\n" % (cell, x_mid, y_mid, bb_w, bb_h)
            f.write(line)
        '''


def find_midpoint(box):
    img_xm = box[0] + (box[2] / 2)
    img_ym = box[1] + (box[3] / 2)

    c_x = (img_xm % CELL_WIDTH) / CELL_WIDTH
    c_y = (img_ym % CELL_HEIGHT) / CELL_HEIGHT

    cell_x = floor(img_xm / CELL_WIDTH)
    cell_y = floor(img_ym / CELL_HEIGHT)

    cell_num = cell_y * GRID_WIDTH + cell_x

    bb_width = box[2] / IMG_WIDTH
    bb_height = box[3] / IMG_HEIGHT

    return cell_num, c_x, c_y, bb_width, bb_height


def rescale_labels(x, y, box_w, box_h, img_w, img_h, pad_top, pad_left):
    s_w = IMG_WIDTH / max(img_w, img_h)
    s_h = IMG_HEIGHT / max(img_w, img_h)

    Nx = s_w * x + pad_left
    Ny = s_h * y + pad_top
    Nw = s_w * box_w
    Nh = s_h * box_h

    return int(Nx), int(Ny), int(Nw), int(Nh)

data = prep_chockpnt_data('C:\\Users\\Humphrey\\Desktop\\chokepoint')
process_data(data['face'], '', '', 'data\\classification\\face')
print('\nface done..')
process_data(data['noface'], '', '', 'data\\classification\\noface')
print('\nnoface done')