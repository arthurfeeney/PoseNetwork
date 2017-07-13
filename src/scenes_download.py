
"""
this script is used to load the 7scenes localization dataset.
It does not load the depth information because I don't need it.
But it should be fairly simple to add that in.
"""

import numpy as np
import pickle
import os
from PIL import Image
import re
import math


# Directory where you want to download and save the data-set.
# Set this before you calling any functions to load data.
data_path = ''

def set_data_path(path):
    global data_path
    data_path = path

img_height = 299
img_width = 399

num_channels = 3

num_classes = 7

_images_per_file = 1000

def _load_data(filename):
    # Load an image
    load_image = Image.open(data_path + filename)

    #rescale the image so the smallest side is 299
    re_image = load_image.resize((399, 299))

    # convert image to a numpy array
    image = np.asarray(re_image)

    # preprocess the image to the range that the model expects. [-1, 1]
    image = 2 * (image / 255.0) - 1.0

    return image

# loads the label as matrix and converts it to a quaternion.
def _load_pls(filename):
    # loads the 4x4 matrix label for cameras pose.
    label_file = open(data_path + filename, 'r')

    file_matrix = label_file.readlines()

    for row in range(len(file_matrix)):
        file_matrix[row] = file_matrix[row].strip()
        file_matrix[row] = re.split(r'\t+', file_matrix[row])

    m = np.array(file_matrix)
    m = m.astype(np.float)

    location = m[0:3, -1] # gets top 3 elem in 4th col.

    q = [0.0 for _ in range(4)]

    # convert rotation matrix to quaternion
    trace = m[0][0] + m[1][1] + m[2][2]

    if trace > 0:
        s = .5 / math.sqrt(trace + 1.0)
        q[0] = .25 / s
        q[1] = (m[2][1] - m[1][2]) * s
        q[2] = (m[0][2] - m[2][0]) * s
        q[3] = (m[1][0] - m[0][1]) * s
    else:
        if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
            q[0] = (m[2][1] - m[1][2]) / s
            q[1] = .25 * s
            q[2] = (m[0][1] + m[1][0]) / s
            q[3] = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
            q[0] = (m[0][2] - m[2][0]) / s
            q[1] = (m[0][1] + m[1][0]) / s
            q[2] = .25 * s
            q[3] = (m[1][2] + m[2][1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
            q[0] = (m[1][0] - m[0][1]) / s
            q[1] = (m[0][2] + m[2][0]) / s
            q[2] = (m[1][2] + m[2][1]) / s
            q[3] = .25 * s

    label = np.concatenate((location, q))

    return label

def load_training_data(scene, size):
    global data_path
    data_path = data_path + scene

    images = np.zeros(
                    shape=[size, img_height, img_width,num_channels],
                    dtype=float
                )
    pls = np.empty(shape=[size, 7], dtype=float)

    with open(data_path + 'TrainSplit.txt') as split_file:
        split = []
        for num in split_file:
            split.append(int(num[-2]) - 1) # -2 because of '\n'.

    begin = 0
    for i in split:
        images_batch = [None]*_images_per_file
        pls_batch = [None]*_images_per_file

        for j in range(_images_per_file):
            zerod_j = str(j).zfill(3)

            im_filename='seq-0'+str(i+1)+'/frame-000'+zerod_j+'.color.png'

            image = _load_data(filename=im_filename)

            pls_filename='seq-0'+str(i+1)+'/frame-000'+zerod_j+'.pose.txt'

            pl = _load_pls(filename=pls_filename)

            images_batch[j] = image

            pls_batch[j] = pl

        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        images[begin:end, :] = images_batch

        pls[begin:end] = pls_batch

        begin = end

    return images, pls

def load_test_data(scene, size):
    global data_path
    data_path = data_path + scene

    images = np.zeros(
                    shape=[size, img_height, img_width,num_channels],
                    dtype=float
                )
    pls = np.empty(shape=[size, 7], dtype=float)

    with open(data_path + 'TestSplit.txt') as split_file:
        split = []
        for num in split_file:
            split.append(int(num[-2]) - 1) # -2 because of '\n'.

    begin = 0
    for i in split:
        images_batch = [None]*_images_per_file
        pls_batch = [None]*_images_per_file

        for j in range(_images_per_file):
            zerod_j = str(j).zfill(3)

            im_filename='seq-0'+str(i+1)+'/frame-000'+zerod_j+'.color.png'

            image = _load_data(filename=im_filename)

            pls_filename='seq-0'+str(i+1)+'/frame-000'+zerod_j+'.pose.txt'

            pl = _load_pls(filename=pls_filename)

            images_batch[j] = image
            pls_batch[j] = pl

        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        images[begin:end, :] = images_batch

        pls[begin:end] = pls_batch

        begin = end
    return images, pls
