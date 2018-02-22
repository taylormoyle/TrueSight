import tensorflow as tf
import numpy as np
from math import sqrt
import Operations as op
from PIL import Image
import glob
import os
import xml.etree.ElementTree as ET

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
def load_data(dir):
    images = []
    path = os.path.join(dir, "*")
    for filename in glob.glob(path):
        im = Image.open(filename)
        img = np.array(im)
        img = img.transpose(2, 0, 1)
        images.append(img)
    return images

# Perform transformations, normalize images, return array of tuples [(norm_image, label)]
def process_data():

