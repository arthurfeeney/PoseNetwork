import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def euclidean_distance(predicted, actual, scale = 6):
    """
    computes the sum of the euclidean distance between the predicted and actual
    position and orientation. -> dist(pos) + scale * dist(ori)
    """
    x, q = tf.split(predicted, [3,4], axis=1)
    x_, q_ = tf.split(actual, [3,4], axis=1)

    b = tf.constant(scale, dtype=tf.float32)

    return tf.norm(x_ - x, asxis=1) + \
           (b * tf.norm(q_ - (q / tf.norm(q, axis=1, keep_dims=True)),axis=1))

# from geometric loss functions paper
def _distance_with_learned_scale(predicted, actual, s_x, s_q):
    """
    loss function learns a scale between position and orientation.
    Uses axis one so that the function is applied to each element of the batch
    and not to the batch itself.
    """
    x, q = tf.split(predicted, [3,4], axis=1)

    x_, q_ = tf.split(actual, [3,4], axis=1)

    loss_pos = tf.add(
                    tf.multiply(
                        tf.norm(tf.subtract(x_, x), axis=1),
                        tf.exp(-s_x)
                    ),
                    s_x
                )

    norm_q = tf.norm(q, axis=1, keep_dims=True)

    unit_q = tf.divide(q, norm_q)

    loss_ori = tf.add(
                tf.multiply(
                    tf.norm(tf.subtract(q_, unit_q), axis=1),
                    tf.exp(-s_q)
                ),
                s_q
            )

    return tf.add(loss_pos, loss_ori)

def _position_and_angle(predicted, actual):
    """
    computes the position and angle error for printing and test accuracy.
    it uses axis 1 to apply function to each element of batch.
    The math functions should all be element wise already.
    """
    x, q = tf.split(predicted, [3,4], axis=1)

    x_, q_ = tf.split(actual, [3,4], axis=1)

    distance = tf.norm(tf.subtract(x_, x), axis=1)

    norm_q = tf.norm(q, axis=1, keep_dims=True)

    unit_q = tf.divide(q, norm_q)

    radian_angle = tf.abs(tf.norm(q_ - unit_q, axis=1))

    angle = tf.divide(tf.multiply(radian_angle, 180.0), np.pi)

    return tf.stack(
                (tf.reduce_mean(distance), tf.reduce_mean(angle)),
                axis=0
           )


def shared_dual_stream(model, lr=1e-3):
    """
    My own network design. Attempt to separate the regression of position
    and orientation into separate streams while still sharing the information
    of what the other stream is doing.
    """
    net = model['PrePool']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    net = slim.conv2d_transpose(model['upper_input'],
                                768,
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d_transpose(net,
                                384,
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    left_1 = slim.conv2d(net,
                         20,
                         (3,3),
                         stride=1,
                         normalizer_fn=slim.batch_norm)

    right_1 = slim.conv2d(net,
                          20,
                          (3,3),
                          stride=1,
                          normalizer_fn=slim.batch_norm)

    left_and_share_right = tf.add(left_1, tf.multiply(right_1, 0.3))

    right_and_share_left = tf.add(right_1, tf.multiply(left_1, 0.3))

    loc_input = slim.flatten(left_and_share_right)

    ori_input = slim.flatten(right_and_share_left)

    loc_1 = slim.fully_connected(loc_input,
                                 512,
                                 activation_fn=None,
                                 normalizer_fn=slim.batch_norm)

    ori_1 = slim.fully_connected(ori_input,
                                 512,
                                 activation_fn=None,
                                 normalizer_fn=slim.batch_norm)

    loc_and_share_ori = tf.add(loc_1, tf.multiply(ori_1, 0.3))

    ori_and_share_loc = tf.add(ori_1, tf.multiply(loc_1, 0.3))

    loc = slim.fully_connected(loc_and_share_ori,
                               3,
                               activation_fn=None,
                               normalizer_fn=slim.batch_norm)

    ori = slim.fully_connected(ori_and_share_loc,
                               4,
                               activation_fn=None,
                               normalizer_fn=slim.batch_norm)

    logits = tf.concat((loc, ori), axis=1)

    s_x = tf.Variable(0.0, dtype=tf.float32, trainable=True)

    s_q = tf.Variable(-3.0, dtype=tf.float32, trainable=True)

    loss = _distance_with_learned_scale(
        predicted=logits,
        actual=model['labels'],
        s_x=s_x,
        s_q=s_q
    )

    distance = _position_and_angle(
        predicted=logits,
        actual=model['labels']
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr
    ).minimize(loss)

    model['acc'] = distance

def decode_dual_stream(model, lr=1e-3):
    """
    Implementation of network from the hourglass networks paper.
    """
    net = model['PrePool']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    net = slim.conv2d_transpose(model['upper_input'],
                                net.get_shape()[3],
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d_transpose(net,
                                768,
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d_transpose(net,
                                384,
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d(net,
                      32,
                      (3,3),
                      stride=1,
                      normalizer_fn=slim.batch_norm)

    net = slim.flatten(net)

    net = slim.fully_connected(net,
                               1024,
                               activation_fn=None,
                               normalizer_fn=slim.batch_norm)

    loc = slim.fully_connected(net,
                               3,
                               activation_fn=None,
                               normalizer_fn=slim.batch_norm)

    ori = slim.fully_connected(net,
                               4,
                               activation_fn=None,
                               normalizer_fn=slim.batch_norm)

    logits = tf.concat((loc, ori), axis=1)

    s_x = tf.Variable(0.0, dtype=tf.float32, trainable=True)

    s_q = tf.Variable(-3.0, dtype=tf.float32, trainable=True)

    loss = _distance_with_learned_scale(
        predicted=logits,
        actual=model['labels'],
        s_x=s_x,
        s_q=s_q
    )

    model['acc'] = _position_and_angle(
        predicted=logits,
        actual=model['labels']
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr
    ).minimize(loss)

