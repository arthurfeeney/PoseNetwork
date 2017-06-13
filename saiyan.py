import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from net import incept_upper_mod, pose_upper
from container import Data
from custom_helper import *
#from cifarDownload import *
from scenesDownload import *
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

    num_classes = 7

    train_images, train_labels = load_training_data()
    test_images, test_labels = load_test_data()

    data = Data((train_images, train_labels, test_images, test_labels))


    # train

    input_tensor = tf.placeholder(tf.float32, shape=[None,480,640,3])

    resize = tf.image.crop_to_bounding_box(
        image=input_tensor,
        offset_height=np.random.randint(low=0,high=480-299),
        offset_width=np.random.randint(low=0,high=640-299),
        target_height=299,
        target_width=299
    )

    feed_tensor = tf.placeholder(
                        tf.float32,
                        shape=[None,image_size,image_size,3])

    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2(
                feed_tensor,
                num_classes=num_classes,
                is_training=True)

    end_points['labels'] = tf.placeholder(tf.float32, shape=[None, 7])

    exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

    pose_upper(end_points, os=7)

    base_out = end_points['PrePool']

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        loader = tf.train.Saver(variables_to_restore)
        loader.restore(sess, checkpoint_file)


        batch_size = 32
        num_epochs = 2
        verbose = True

        for i in range(num_epochs):
            for index in range(0, data.train_size()-batch_size-1, batch_size):
                data.shuffle()
                images, labels = data.get_next_train_batch(batch_size)

                if verbose and index % (2 * batch_size) == 0:
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
                    print('step: ' + str(index) + ' acc: ' + str(batch_acc))
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
        saver.save(sess,
                   '/data/zhanglab/afeeney/cifar_model/cifar_test',
                   global_step=0)

    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(
            '/data/zhanglab/afeeney/cifar_model/cifar_test-0.meta'
        )
        loader.restore(sess, tf.train.latest_checkpoint(
            '/data/zhanglab/afeeney/cifar_model'
        ))
        test_images, test_labels = data.test_set()

        batch_size = 1
        acc = 0
        count = 0

        for index in range(len(test_labels)-batch_size-1):
            acc += end_points['acc'].eval(
                feed_dict = {
                    end_points['upper_input']: base_out.eval(
                        feed_dict = {
                            feed_tensor: resize.eval(
                                feed_dict = {
                                    input_tensor: test_images[index:index+1]
                                }
                            )
                        }
                    ),
                    end_points['labels']: test_labels[index:index+1]
                }
            )
            count += 1

        print(acc / count)

    print('finished')

if __name__ == "__main__":
    main()
