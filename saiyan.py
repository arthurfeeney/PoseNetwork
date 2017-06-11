import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from net import incept_upper_mod, cifar_base_net, cifar_upper_net, create_base_net
from container import Data
from custom_helper import *
from cifarDownload import *
from inception_resnet_v2 import *

slim = tf.contrib.slim

# Hopefully, this will eventually become a fully-functional implementation
# of GoogleNet! Which will then be modified to a variant of PoseNet.

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
 	    'step': step,
	    'acc': accuracy
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


    trainImages, trainLabels = load_training_data()
    testImages, testLabels = load_test_data()

    data = Data((trainImages, trainLabels, testImages, testLabels))

    with tf.Session() as sess:

        input_tensor = tf.placeholder(tf.float32, shape=[None,32,32,3])

        resize = tf.image.resize_images(input_tensor, [299,299])

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
        #saver.save(sess, './poop', global_step=0)
        #loader = tf.train.Saver(variables_to_restore)
        sess.run(tf.global_variables_initializer())
        #loader = tf.train.import_meta_graph('./poop-0.meta')
        loader = tf.train.Saver(variables_to_restore)
        loader.restore(sess, checkpoint_file)

        base_out = end_points['PrePool']

        batch_size = 32

        for index in range(0, len(data.labels())-batch_size-1, batch_size):
            data.shuffleData()
            batch = data.trainBatch(batch_size)
            #batch[0] = 2*(images/299.0)-1.0



            if index % (2 * batch_size) == 0:
                batch_acc = end_points['acc'].eval(
                    feed_dict = {
                        end_points['upper_input']: base_out.eval(
                            feed_dict = {
                                feed_tensor: resize.eval(
                                    feed_dict = {
                                        input_tensor: batch[0]
                                    }
                                )
                            }
                        ),
                        end_points['labels']: batch[1]
                    }
                )
                print(batch_acc)

            end_points['step'].run(
                feed_dict = {
                    end_points['upper_input']: base_out.eval(
                        feed_dict = {
                            feed_tensor: resize.eval(
                                feed_dict = {
                                    input_tensor: batch[0]
                                }
                            )
                        }
                    ),
                    end_points['labels']: batch[1]
                }
            )
    print('finished')

def train(model, data, save_weight_file, init_weight_file=None,
          batch_size=32, epoch=1, verbose=False):
    with tf.Session() as sess:
        if init_weight_file is not None:
            loader = tf.train.import_meta_graph(init_weight_file)
            loader.restore(sess, tf.train.latest_checkpoint(
                '/data/zhanglab/afeeney/cifar_model/'
            ))
        else:
            sess.run(tf.global_variables_initializer())

        for e in range(epoch):
            data.shuffleData()
            # modifying number of steps.
            for epi in range(0, data.trainSize()-batch_size-1, batch_size):
                ib, lb = data.trainBatch(batch_size)
                if verbose and epi % (20*batch_size) == 0:
                    # this should use validation set for ImageNet
                    # keeping score of the last validation set.
                    # also after the training part.
                    batchAccuracy = model['acc'].eval(
                        feed_dict = {
                            model['images']: ib,
                            model['labels']: lb
                        }
                    )
                    print('step %d, acc %g'%(epi, batchAccuracy))
                model['step'].run(
                    feed_dict = {
                        model['images']: ib,
                        model['labels']: lb
                    }
                )
            saver = tf.train.Saver()
            saver.save(sess, save_weight_file, global_step=e)

def mod_train(model, data, save_weight_file,
              init_weight_file, batch_size=32, epoch=1, verbose=False):
    with tf.Session() as sess:

        model = cifar_upper_net(model)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        loader = tf.train.import_meta_graph(init_weight_file)
        loader.restore(sess, tf.train.latest_checkpoint(
            '/data/zhanglab/afeeney/cifar_model/'
        ))

        model['base_out'] = \
            tf.get_default_graph().get_tensor_by_name('base_output1:0')

        for e in range(epoch):
            data.shuffleData()
            for epi in range(0, data.trainSize()-batch_size-1, batch_size):
                batch = data.trainBatch(batch_size)
                model['step'].run(
                    feed_dict = {
                        model['upper_input']: model['base_out'].eval(
                            feed_dict = {
                                model['images']: batch[0],
                                model['labels']: batch[1]
                            }
                        ),
                        model['labels']: batch[1]
                    }
                )
        saver.save(sess, save_weight_file, global_step=0)
    return model

def mod_test(model, data, weight_file, verbose=False, batch_size=16):
    print('MODIFY_test called')
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(weight_file)
        loader.restore(sess, tf.train.latest_checkpoint(
            '/data/zhanglab/afeeney/cifar_model/'
        ))

        images, labels = data.test()
        acc = 0
        count = 0

        for index in range(len(labels)-batch_size-1):
            if verbose and index % (10 * batch_size) == 0:
                print('tested: %d images'%index)
            acc += model['acc'].eval(
                feed_dict = {
                    model['upper_input']: model['base_out'].eval(
                        feed_dict = {
                            model['images']: images[index:index+batch_size],
                            model['labels']: labels[index:index+batch_size]
                        }
                    ),
                    model['labels']: labels[index:index+batch_size]
                }
            )
            count += 1

        test_acc = acc / count
        print('test acc: %g'%test_acc)
    return False


def test(model, data, weight_file, verbose=False):
    print('test called')
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(weight_file)
        loader.restore(sess, tf.train.latest_checkpoint(
            '/data/zhanglab/afeeney/cifar_model/'
        ))

        images, labels = data.test()
        acc = 0
        count = 0

        batch_size = 16 # small default so it is a bit smoother.
        for index in range(len(labels)-batch_size-1):
            if verbose and index % (10 * batch_size) == 0:
                print('tested: %d images'%index)
            acc += model['acc'].eval(
                feed_dict = {
                    model['images']: images[index:index+batch_size],
                    model['labels']: labels[index:index+batch_size]
                }
            )
            count += 1
        test_acc = acc / count
        print('test acc: %g'%test_acc)
    return False

if __name__ == "__main__":
    main()
