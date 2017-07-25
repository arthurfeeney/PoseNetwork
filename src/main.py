import tensorflow as tf
import numpy as np
from net import shared_dual_stream, decode_dual_stream, simple_stream
from data_class import Data
from scenes_download import *
from inception_resnet_v2 import *

def main():

    checkpoint_file = \
        '/data/zhanglab/afeeney/inception_resnet_v2_2016_08_30.ckpt'

    scenes_path = '/data/zhanglab/afeeney/7scenes/'

    scene = 'heads/'

    image_size = 299

    num_classes = 7

    set_data_path(scenes_path)
    train_data = load_training_data(scene, 1000)
    set_data_path(scenes_path)
    test_data = load_test_data(scene, 1000)

    data = Data(train_data, test_data)

    print('starting training')

    save_location = '/data/zhanglab/afeeney/checkpoints/'
    scene_name = scene[0:-2]

    train(checkpoint_file,
          save_location,
          scene_name,
          data,
          batch_size=40,
          num_epochs=60,
          verbose=True)

    print('starting testing')

    distance_error, angle_error = test(data, save_location)

    print('finished testing')

    print('distance: ' + str(distance_error) + ' angle: ' + str(angle_error))

def feed_helper(end_points,
                images,
                labels,
                is_training=True,
                keep_prob=0.5,
                random_crop=True,
                image_size=299):

    offset_height, offset_width = 0, 0

    if random_crop:
        offset_width=np.random.randint(low=0, high=399-image_size)
    else:
        offset_width=int((399-image_size)/2)

    resize = tf.image.crop_to_bounding_box(
        image=end_points['input_tensor'],
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=image_size,
        target_width=image_size
    )

    return {
        end_points['upper_input']: end_points['Conv2d_7b_1x1'].eval(
            feed_dict = {
                end_points['feed_tensor']: resize.eval(
                    feed_dict = {
                        end_points['input_tensor']: images
                    }
                )
            }
        ),
        end_points['labels']: labels,
        end_points['keep_prob']: keep_prob,
        end_points['is_training']: is_training
    }

def train(checkpoint_file,
          save_location,
          scene_name,
          data,
          image_size=299,
          num_epochs=1,
          batch_size=32,
          verbose=False):
    with tf.device('/gpu:0'):
        input_tensor = tf.placeholder(tf.float32, shape=[None,299,399,3])

        feed_tensor = tf.placeholder(tf.float32,
                                     shape=[None,image_size,image_size,3])

        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_resnet_v2(feed_tensor,
                                                     num_classes=num_classes,
                                                     is_training=True)

        end_points['labels'] = tf.placeholder(tf.float32, shape=[None, 7])

        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        end_points['input_tensor'] = input_tensor
        end_points['feed_tensor'] = feed_tensor

        update = shared_dual_stream(end_points)
        #update = decode_dual_stream(end_points)
        #update = simple_stream(end_points)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        loader = tf.train.Saver(variables_to_restore)
        loader.restore(sess, checkpoint_file)

        for epoch in range(num_epochs):
            for step in range(0, data.train_size(), batch_size):
                if epoch != 0:
                    data.shuffle()

                images, labels = data.get_next_batch(batch_size)

                if verbose and step % (20 * batch_size) == 0:
                    batch_acc = end_points['acc'].eval(
                        feed_dict=feed_helper(end_points, images, labels))
                    print('epoch: ' + str(epoch) + ' step: ' + \
                          str(step) + ' acc: ' + str(batch_acc))

                update.run(feed_dict=feed_helper(end_points, images, labels))

        saver.save(sess,
                   (save_location + scene_name),
                   global_step=0)

def test(data,
         load_location,
         image_size=299,
         batch_size=1,
         verbose=False):
    loader = tf.train.Saver()
    with tf.device('/gpu:0'):
        input_tensor = tf.placeholder(tf.float32, shape=[None,299,399,3])

        feed_tensor = tf.placeholder(tf.float32,
                                     shape=[None,image_size,image_size,3])

        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_resnet_v2(feed_tensor,
                 num_classes=num_classes, is_training=False, reuse=True)

        end_points['labels'] = tf.placeholder(tf.float32, shape=[None, 7])

        exclude = ['InceptionResnetV2/Logits',
                  'InceptionResnetV2/AuxLogits']
        variables_to_restore = \
            slim.get_variables_to_restore(exclude=exclude)

        end_points['input_tensor'] = input_tensor
        end_points['feed_tensor'] = feed_tensor
        _ = shared_dual_stream(end_points, reuse=True)
        #_ = decode_dual_stream(end_points, reuse=True)
        #_ = simple_stream(end_points, reuse=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        loader.restore(sess, tf.train.latest_checkpoint(load_location))

        acc = np.array([0,0], dtype=np.float32)
        count = 0

        data.reset_batch()


        for i in range(0, data.test_size(), batch_size):
            count += batch_size

            images, labels = data.get_next_batch(batch_size, get_test=True)

            acc += end_points['acc'].eval(
                feed_dict=feed_helper(end_points, images, labels,
                                      random_crop=False, image_size=image_size,
                                      keep_prob=.5, is_training=False))

            if verbose:
                print('count: ' + str(i) + 'acc: ' + str(acc))

    ave_test_acc = acc / count
    return ave_test_acc

if __name__ == "__main__":
    main()
