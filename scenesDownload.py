
# pretty heavily modified version of cifarDownload.

import numpy as np
import pickle
import os
from PIL import Image
import download
import re
import math
#from dataset import one_hot_encoded

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "/data/zhanglab/afeeney/7scenes/chess/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_height = 480
img_width = 640

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_height * img_width * num_channels

# Number of classes.
num_classes = 7

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 4

# Number of images for each batch-file in the training-set.
_images_per_file = 1000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, "chess/", filename)

def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_height, img_width])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    # Load an image
    load_image = Image.open(data_path + filename)

    image = np.asarray(load_image)

    return image

# loads the label as matrix and converts it to a quaternion.
def _load_cls(filename):
    # loads the 4x4 matrix label for cameras pose.
    label_file = open(data_path + filename, 'r')

    file_matrix = label_file.readlines()

    for row in range(len(file_matrix)):
        file_matrix[row] = file_matrix[row].strip()
        file_matrix[row] = re.split(r'\t+', file_matrix[row])

    m = np.array(file_matrix)
    m = m.astype(np.float)

    location = m[0:3, -1] # gets top 3 elem in 4th col.
    quaternion = [0.0]*4

    quaternion[0] = math.sqrt(1 + m[0][0] + m[1][1] + m[2][2]) / 2.0
    q4 = 4.0 * quaternion[0]
    quaternion[1] = (m[2][1] - m[1][2]) / q4
    quaternion[2] = (m[0][2] - m[2][0]) / q4
    quaternion[3] = (m[1][0] - m[0][1]) / q4

    label = np.concatenate((location, quaternion))

    return label


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def load_training_data():
    images = np.zeros(
                    shape=[_num_images_train, img_height,
                           img_width,num_channels],
                    dtype=float
                )
    pls = np.empty(shape=[_num_images_train, 7], dtype=float)
    begin = 0
    for i in [0,1,3,5]:
        print(i)
        images_batch = [None]*_images_per_file
        pls_batch = [None]*_images_per_file
        for j in range(_images_per_file):
            zerod_j = str(j).zfill(3)
            im_filename='seq-0'+str(i+1)+'/frame-000'+zerod_j+'.color.png'
            image = _load_data(filename=im_filename)

            pls_filename='seq-0'+str(i+1)+'/frame-000'+zerod_j+'.pose.txt'

            pl = _load_cls(filename=pls_filename)

            images_batch[j] = image
            #images = np.append(images, image)
            pls_batch[j] = pl

        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        pls[begin:end] = pls_batch

        begin = end

    return images, pls

def load_test_data():
    images = np.zeros(
                    shape=[2000, img_height,
                           img_width,num_channels],
                    dtype=float
                )
    pls = np.empty(shape=[2000, 7], dtype=float)
    begin = 0
    for i in [2, 4]:
        images_batch = [None]*_images_per_file
        pls_batch = [None]*_images_per_file
        for j in range(_images_per_file):
            zerod_j = str(j).zfill(3)
            im_filename='seq-0'+str(i+1)+'/frame-000'+zerod_j+'.color.png'
            image = _load_data(filename=im_filename)

            pls_filename='seq-0'+str(i+1)+'/frame-000'+zerod_j+'.pose.txt'

            pl = _load_cls(filename=pls_filename)

            images_batch[j] = image
            pls_batch[j] = pl

        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        pls[begin:end] = pls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end
    return images, pls
