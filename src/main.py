import tensorflow as tf
import numpy as np
from net import shared_dual_stream, decode_dual_stream
from container import Data
from scenesDownload import *
from inception_resnet_v2 import *

def main():

    checkpoint_file = \
        '/data/zhanglab/afeeney/inception_resnet_v2_2016_08_30.ckpt'

    image_size = 299

    num_classes = 7

    set_data_path('/data/zhanglab/afeeney/7scenes/')
    train_data = load_training_data('chess/', 4000)
    set_data_path('/data/zhanglab/afeeney/7scenes/')
    test_data = load_test_data('chess/', 2000)

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

        shared_dual_stream(end_points, lr=1e-4)

        end_points['input_tensor'] = input_tensor
        end_points['feed_tensor'] = feed_tensor


    print('starting training')

    train(end_points,
          variables_to_restore,
          '/data/zhanglab/afeeney/inception_resnet_v2_2016_08_30.ckpt',
          data,
          batch_size=32,
          num_epochs=90,
          verbose=True)

    print('finished training')

    print('starting testing')

    error = test(end_points, data)

    print('finished testing')

    print('distance: ' + str(error[0]) + ' angle: ' + str(error[1]))

def feed_helper(end_points,
                images,
                labels,
                random_crop=True,
                image_size=299):

    offset_height, offset_width = 0, 0
    if random_crop:
        offset_height=np.random.randint(low=0, high=480-image_size)
        offset_width=np.random.randint(low=0, high=640-image_size)
    else:
        offset_height=int((480-image_size)/2)
        offset_width=int((640-image_size)/2)
    resize = tf.image.crop_to_bounding_box(
        image=end_points['input_tensor'],
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=image_size,
        target_width=image_size
    )
    return {
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

        for epoch in range(num_epochs):
            for step in range(0, data.train_size()-batch_size-1, batch_size):
                data.shuffle()

                images, labels = data.get_next_batch(batch_size)

                if verbose and step % (20 * batch_size) == 0:
                    batch_acc = end_points['acc'].eval(
                        feed_dict=feed_helper(
                                    end_points,
                                    images,
                                    labels)
                    )
                    print('epoch: ' + str(epoch) + ' step: ' + \
                          str(step) + ' acc: ' + str(batch_acc))

                end_points['step'].run(
                    feed_dict=feed_helper(end_points, images, labels)
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
        count = 0

        for i in range(data.test_size()-batch_size-1):
            count += 1

            images, labels = data.get_next_batch(batch_size, get_test=True)

            acc += end_points['acc'].eval(
                feed_dict=feed_helper(
                    end_points,
                    images,
                    labels,
                    random_crop=False,
                    image_size=image_size
                )
            )

            if verbose:
                print('count: ' + str(i) + 'acc: ' + str(acc))

    return acc / count

if __name__ == "__main__":
    main()
