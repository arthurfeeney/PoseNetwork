import numpy as np
import random

class Data:
    def __init__(self, train_data, test_data):
        self.__train_images = train_data[0]
        self.__train_labels = train_data[1]
        self.__test_images = test_data[0]
        self.__test_labels = test_data[1]
        self.__last_batch_end = 0

    # change size of range in the inner list to 1000
    # don't need to do one-hot encoding for 7 scenes.
    def __label_to_vect(self, labels):
        return [[0 if a != label else 1 for a in range(7)] for label in labels]

    def remain(self):
        return len(self.__train_labels) - self.__last_batch_end

    def train_size(self):
        return len(self.__train_labels)

    def get_next_train_batch(self, batch_size):
        batch_indices = \
            np.array(
                range(self.__last_batch_end, self.__last_batch_end+batch_size)
            )
        self.__last_batch_end += batch_size
        return (
            self.__train_images[batch_indices[0]:batch_indices[-1]],
            self.__train_labels[batch_indices[0]:batch_indices[-1]]
        )

    def test_set(self):
        return self.__test_images, self.__test_labels

    # shuffles images and labels together. also resets the index of last batch.
    def shuffle(self, shuffle_test = False):
        self.__last_batch_end = 0
        c = list(zip(self.__train_images, self.__train_labels))
        random.shuffle(c)
        self.__train_images, self.__train_labels = zip(*c)
        if shuffle_test:
            c = list(zip(self.__test_images, self.__test_labels))
            random.shuffle(c)
            self.__test_images, self.__test_labels = zip(*c)
