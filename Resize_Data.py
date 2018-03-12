import random
from math import floor
from PIL import Image
import os
import numpy as np
import random

IMG_HEIGHT = 208
IMG_WIDTH = 208
CELL_HEIGHT = 16
CELL_WIDTH = 16
GRID_HEIGHT = 13
GRID_WIDTH = 13
RES = 208

def prep_face_data(data_directory):
    dataset = {'training': [], 'test': [], 'validation': []}
    labels = {}
    for datas in dataset:
        with open(data_directory + '\\resized_face_images\\annotations_' + datas + '.txt') as f:
            filename = ""
            count = 0
            line = f.readline()
            while not line == "":
                if ".jpg" in line:
                    filename = os.path.join(data_directory, line)
                    count = int(f.readline())
                label = np.zeros((169, 5))
                for b in range(count):
                    box = f.readline().split()
                    cell = int(box[0])
                    x = float(box[1])
                    y = float(box[2])
                    w = float(box[3])
                    h = float(box[4])
                    if np.array_equal(label[cell], [0, 0, 0, 0, 0]):
                        label[cell] += [1, x, y, w, h]
                if count <= 10:
                    dataset[datas].append(filename[:-1])
                    labels[filename[:-1]] = label
                line = f.readline()
            print(label)
        random.shuffle(dataset[datas])
    return dataset, labels


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
    img = 0
    set_annotations_file = os.path.join(save_dir, "annotations_%s.txt" % data_set)
    with open(set_annotations_file, 'w') as f:
        for i in data:
            f.write(i + "\n")

            # load images, resize, and save resized images in save_dir
            im = Image.open(i)
            og_w, og_h = im.size
            img, pad_top, pad_left = resize_img(im)
            image = Image.fromarray(img)

            img_folder, filename = os.path.split(i)
            _, img_folder = os.path.split(img_folder)

            img_save_dir = os.path.join(save_dir, data_set, img_folder)
            if not os.path.exists(img_save_dir):
                os.mkdir(img_save_dir)

            file_pathname = os.path.join(img_save_dir, filename)
            image.save(file_pathname)

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


dataset, labels = prep_face_data('C:\\Users\\Shadow\\PycharmProjects\\TrueSight\\data')

