import numpy as np
import random

class Data:
    def __init__(self, data):
        self.__trainImages = np.array(data[0])
        self.__trainLabels = np.array(self.__labelToVect(data[1]))
        self.__testImages = np.array(data[2])
        self.__testLabels = np.array(self.__labelToVect(data[3]))
        self.__lastBatchEnd = 0

    # change size of range in the inner list to 1000
    def __labelToVect(self, labels):
        return [[0 if a != label else 1 for a in range(10)] for label in labels]

    def remain(self):
        return len(self.__trainLabels) - self.__lastBatchEnd

    def trainSize(self):
        return len(self.__trainLabels)

    def trainBatch(self, batchSize):
        batchIndices = \
            np.array(range(self.__lastBatchEnd, self.__lastBatchEnd+batchSize))
        self.__lastBatchEnd += batchSize
        return (
            self.__trainImages[batchIndices], self.__trainLabels[batchIndices]
        )

    def test(self):
        return self.__testImages, self.__testLabels

    # shuffles images and labels together. also resets the index of last batch.
    def shuffleData(self, shuffleTest = False):
        self.__lastBatchEnd = 0
        if not shuffleTest:
            c = list(zip(self.__trainImages, self.__trainLabels))
            random.shuffle(c)
            a, b = zip(*c)
            self.__trainImages, self.__trainLabels = np.array(a), np.array(b)
        else:
            c = list(zip(self.__testImages, self.__testLabels))
            random.shuffle(c)
            self.__testImages, self.__testLabels = zip(*c)
