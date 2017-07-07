import numpy as np
import random

class Data:
    def __init__(self, train_data, test_data):
        self.__train_images = train_data[0]
        self.__train_labels = train_data[1]
        self.__test_images = test_data[0]
        self.__test_labels = test_data[1]
        self.__last_batch_end = 0
        self.__num_train_images = len(self.__train_labels)
        self.__num_test_images = len(self.__test_labels)

    def train_size(self):
        return self.__num_train_images

    def test_size(self):
        return self.__num_test_images

    def reset_batch(self):
        self.__last_batch_end = 0

    def get_next_batch(self, batch_size, get_test=False):
        batch_indices = \
            np.array(
                range(self.__last_batch_end,
                      self.__last_batch_end+batch_size+1)
            )

        if not get_test:
            self.__last_batch_end += batch_size
            return (
                self.__train_images[batch_indices[0]:batch_indices[-1]],
                self.__train_labels[batch_indices[0]:batch_indices[-1]]
            )

        else:
            self.__last_batch_end += batch_size
            return (
                self.__test_images[batch_indices[0]:batch_indices[-1]],
                self.__test_labels[batch_indices[0]:batch_indices[-1]]
            )

    # shuffles images and labels together. also resets the index of last batch.
    def shuffle(self, shuffle_test = False):
        self.__last_batch_end = 0

        if not shuffle_test:
            c = list(zip(self.__train_images, self.__train_labels))
            random.shuffle(c)
            self.__train_images, self.__train_labels = zip(*c)

        elif shuffle_test:
            c = list(zip(self.__test_images, self.__test_labels))
            random.shuffle(c)
            self.__test_images, self.__test_labels = zip(*c)
