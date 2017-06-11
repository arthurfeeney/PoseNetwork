import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from net import incept_upper_mod
from container import Data
from custom_helper import *
from cifarDownload import *
from inception_resnet_v2 import *
from train import train, mod_train
from test import test, mod_test

slim = tf.contrib.slim

def main():

    set_batch_count()
    """
    trainImages, trainLabels = load_training_data()
    testImages, testLabels = load_test_data()

    cifarData = Data((trainImages, trainLabels, testImages, testLabels))

    images, labels, step, accuracy = cifar_base_net(.001, 10)

    model = {
        'images': images,
        'labels': labels,
 	    'step': step,	    'acc': accuracy
    }

    #cifar_dir = '/data/zhanglab/afeeney/cifar_model/'
    #cifar_save_path = '/data/zhanglab/afeeney/cifar_model/cifar_test'
    # modify load_path based on the number of epochs of last training.
    #cifar_load_path = '/data/zhanglab/afeeney/cifar_model/cifar_test-0.meta'

    train(model, cifarData, save_weight_file=cifar_save_path, batch_size=16)

    model = mod_train(
        model,
        cifarData,
        save_weight_file=cifar_save_path,
        init_weight_file=cifar_load_path,
        epoch = 1
    )

    mod_test(model, cifarData, weight_file=cifar_load_path, verbose=True)
    """

    checkpoint_file = \
        '/data/zhanglab/afeeney/inception_resnet_v2_2016_08_30.ckpt'

    image_size = 299

    num_classes = 10


    train_images, train_labels = load_training_data()
    test_images, test_labels = load_test_data()

    data = Data((train_images, train_labels, test_images, test_labels))

    with tf.Session() as sess:

        input_tensor = tf.placeholder(tf.float32, shape=[None,32,32,3])

        resize = tf.image.resize_images(input_tensor,[299,299])

        feed_tensor = tf.placeholder(tf.float32, shape=[None,299,299,3])

        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_resnet_v2(
                    feed_tensor,
                    num_classes=10,
                    is_training=True)

        end_points['labels'] = tf.placeholder(tf.float32, shape=[None, 10])

        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

        incept_upper_mod(end_points)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        loader = tf.train.Saver(variables_to_restore)
        loader.restore(sess, checkpoint_file)

        base_out = end_points['PrePool']

        batch_size = 32

        for index in range(0, data.train_size()-batch_size-1, batch_size):
            data.shuffle()
            images, labels = data.get_next_train_batch(batch_size)
            images[:,:,:] = 2*(images/299.0)-1.0


            if index % (2 * batch_size) == 0:
                batch_acc = end_points['acc'].eval(
                    feed_dict = {
                        end_points['upper_input']: base_out.eval(
                            feed_dict = {
                                feed_tensor: resize.eval(
                                    feed_dict = {
                                        input_tensor: images
                                    }
                                )
                            }
                        ),
                        end_points['labels']: labels
                    }
                )
                print(batch_acc)

            end_points['step'].run(
                feed_dict = {
                    end_points['upper_input']: base_out.eval(
                        feed_dict = {
                            feed_tensor: resize.eval(
                                feed_dict = {
                                    input_tensor: images
                                }
                            )
                        }
                    ),
                    end_points['labels']: labels
                }
            )


        test_images, test_labels = data.test_set()




    print('finished')

if __name__ == "__main__":
    main()
