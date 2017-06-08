import tensorflow as tf
import numpy as np
from net import cifar_base_net, create_base_net
from container import Data
from cifarDownload import *

# Hopefully, this will eventually become a fully-functional implementation
# of GoogleNet! Which will then be modified to a variant of PoseNet.

def main():
    trainImages, trainLabels = load_training_data()
    testImages, testLabels = load_test_data()
    cifarData = Data((trainImages, trainLabels, testImages, testLabels))
    images, labels, step, accuracy = create_base_net(.001, 10)
    model = {'images': images, 'labels': labels, 'step': step, 'acc': accuracy}

    #
    # When training on imagenet, don't pass init_weight_file.
    # Pass init_weight_file ImageNet to use trained ImageNet base model.
    #

    train(model, cifarData, save_weight_file='./cifar_test', batch_size=16)
    #train(
    #      model, cifarData, save_weight_file='./cifar_test',
    #      init_weight_file='./cifar_test-0.meta',
    #      batch_size=32, epoch=2
    #     )

    test(model, cifarData, weight_file='./cifar_test-0.meta', verbose=True)

    print 'finished'

def train(model, data, save_weight_file, init_weight_file=None,
          batch_size=32, epoch=1, verbose=True):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if init_weight_file is not None:
            loader = tf.train.import_meta_graph(init_weight_file)
            loader.restore(sess, tf.train.latest_checkpoint('./'))
        else:
            sess.run(tf.global_variables_initializer())
        for e in xrange(epoch):
            data.shuffleData()
            for epi in range(0, data.trainSize(), batch_size):
                ib, lb = data.trainBatch(batch_size)
                if verbose and epi % (5*batch_size) == 0:
                    # this should use validation set for ImageNet
                    # keeping score of the last validation set.
                    batchAccuracy = model['acc'].eval(
                        feed_dict = {
                            model['images']: ib,
                            model['labels']: lb
                        }
                    )
                    print 'step %d, acc %g'%(epi, batchAccuracy)
                model['step'].run(
                    feed_dict = {
                        model['images']: ib,
                        model['labels']: lb
                    }
                )
            saver.save(sess, save_weight_file, global_step=e)

# only going to be testing on poses.
# may work better if the batch size is smaller.
def test(model, data, weight_file, verbose=False):
    print 'test called'
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(weight_file)
        loader.restore(sess, tf.train.latest_checkpoint('./'))
        images, labels = data.test()
        acc = 0
        count = 0
        for index in xrange(len(labels)-33):
            if verbose and index % (10 * 32) == 0:
                print 'test count: %d'%index
            acc += model['acc'].eval(
                feed_dict = {
                    model['images']: images[index:index+32],
                    model['labels']: labels[index:index+32]
                }
            )
            count += 1
        test_acc = acc / count
        print 'test acc: %g'%test_acc
    return False

if __name__ == "__main__":
    main()
