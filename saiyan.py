import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from net import decode_dual_stream
from container import Data
from custom_helper import *
from scenesDownload import *
from inception_resnet_v2 import *

slim = tf.contrib.slim

def main():

    set_batch_count() # just used for nameing.

    checkpoint_file = \
        '/data/zhanglab/afeeney/inception_resnet_v2_2016_08_30.ckpt'

    image_size = 299

    num_classes = 7

    train_data = load_training_data()
    test_data = load_test_data()

    data = Data(train_data, test_data)

    with tf.device('/gpu:0'):
        input_tensor = tf.placeholder(
                        tf.float32,
                        shape=[None,480,640,3])


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

        decode_dual_stream(end_points)

        end_points['input_tensor'] = input_tensor
        end_points['feed_tensor'] = feed_tensor

    print('starting training')

    train(end_points,
          variables_to_restore,
          '/data/zhanglab/afeeney/inception_resnet_v2_2016_08_30.ckpt',
          data,
          batch_size=32,
          num_epochs=90,
          verbose=False)

    print('testing is already done!')

    print('starting testing')

    error = test(end_points, data)

    print('distance' + str(error[0]) + ' angle ' + str(error[1]))

    print('finished')

def train(end_points,
          variables_to_restore,
          checkpoint_file,
          data,
          image_size=299,
          num_epochs=1,
          batch_size=32,
          verbose=False):
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        loader = tf.train.Saver(variables_to_restore)
        loader.restore(sess, checkpoint_file)

        for i in range(num_epochs):
            for index in range(0, data.train_size()-batch_size-1, batch_size):
                data.shuffle()

                images, labels = data.get_next_batch(batch_size)

                # random crop
                resize = tf.image.crop_to_bounding_box(
                    image=end_points['input_tensor'],
                    offset_height=np.random.randint(low=0, high=480-299),
                    offset_width=np.random.randint(low=0, high=640-299),
                    target_height=image_size,
                    target_width=image_size
                )

                if verbose and index % (2 * batch_size) == 0:
                    batch_acc = end_points['acc'].eval(
                        feed_dict = {
                            end_points['upper_input']:\
                                end_points['PrePool'].eval(feed_dict = {
                                    end_points['feed_tensor']: resize.eval(
                                        feed_dict = {
                                            end_points['input_tensor']:images
                                        }
                                    )
                                }
                            ),
                            end_points['labels']: labels
                        }
                    )

                    print('step: '+str(index)+' acc: '+str(batch_acc))

                end_points['step'].run(
                    feed_dict = {
                        end_points['upper_input']: end_points['PrePool'].eval(
                            feed_dict = {
                                end_points['feed_tensor']: resize.eval(
                                    feed_dict = {
                                        end_points['input_tensor']: images
                                    }
                                )
                            }
                        ),
                        end_points['labels']: labels
                    }
                )

        saver.save(sess,
                   '/data/zhanglab/afeeney/chess_test',
                   global_step=0)

def test(end_points,
         data,
         image_size=299,
         batch_size=1,
         verbose=False):
    with tf.Session() as sess:
        # meta-graph location should be passed in.
        loader = tf.train.import_meta_graph(
            '/data/zhanglab/afeeney/chess_test-0.meta'
        )
        loader.restore(sess, tf.train.latest_checkpoint(
            '/data/zhanglab/afeeney/'
        ))

        # acc and count are used to average for the test accuracy.
        acc = np.array([0,0], dtype=np.float32)

        # center crop
        resize = tf.image.crop_to_bounding_box(
            image=end_points['input_tensor'],
            offset_height=int((480-image_size)/2),
            offset_width=int((640-image_size)/2),
            target_height=image_size,
            target_width=image_size
        )

        for i in range(data.test_size()-batch_size-1):
            images, labels = data.get_next_batch(batch_size, get_test=True)

            acc += end_points['acc'].eval(
                feed_dict = {
                    end_points['upper_input']: end_points['PrePool'].eval(
                        feed_dict = {
                            end_points['feed_tensor']: resize.eval(
                                feed_dict = {
                                    end_points['input_tensor']: images
                                }
                            )
                        }
                    ),
                    end_points['labels']: labels
                }
            )
            if verbose:
                print('count: ' + str(i) + 'acc: ' + str(acc))

    #print('test acc: ' + str(acc / num_test_images))
    return acc / (data.test_size()-1)

if __name__ == "__main__":
    main()
