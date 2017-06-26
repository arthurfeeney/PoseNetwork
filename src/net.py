import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def euclidean_distance(predicted, actual, scale = 10):
    x, q = tf.split(predicted, [3,4], axis=0)
    x_, q_ = tf.split(actual, [3,4], axis=0)

    b = tf.constant(scale, dtype=tf.float32)

    return tf.norm(x_ - x) + \
           tf.multiply(b, tf.norm(q_ - tf.divide(q, tf.norm(q))))

# from geometric loss functions paper
def distance_with_learned_scale(predicted, actual):
    """
    loss function learns a scale between position and orientation.
    Use axis one so that the function is applied to each element of the batch
    and not to the batch itself.
    """
    position, orientation = tf.split(predicted, [4,5], axis=1)

    x, s_x = tf.split(position, [3,1], axis=1)

    q, s_q = tf.split(orientation, [4,1], axis=1)

    x_, q_ = tf.split(actual, [3,4], axis=1)

    left = tf.add(tf.multiply(tf.norm(tf.subtract(x_, x), axis=1),
                    tf.exp(-s_x)), s_x)

    norm_q = tf.norm(q, axis=1, keep_dims=True)

    unit_q = tf.divide(q, norm_q)

    #arr = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)

    #l_q = arr.unstack(q)
    #q_norms = [tf.norm(qwop) for qwop in l_q]
    #unit_q = l_q.stack(q / q_norms)

    right = tf.add(
                tf.multiply(
                    tf.norm(tf.subtract(q_, unit_q), axis=1),
                    tf.exp(-s_q)
                ),
                s_q
            )

    return tf.add(left, right)

def _get_angle(q):
    qr, rest = tf.split(q, [1,3], axis=1)

    left_arg = tf.sqrt(tf.reduce_sum(tf.square(rest)))

    angle = tf.multiply(2.0, tf.atan2(left_arg, qr))

    return angle

def position_and_angle(predicted, actual):
    """
    computes the position and angle error for printing.
    it uses axis 1 to apply function to each element of batch.
    The math functions should all be element wise already.
    """
    position, orientation = tf.split(predicted, [4,5], axis=1)

    x, _ = tf.split(position, [3,1], axis=1)

    q, _ = tf.split(orientation, [4,1], axis=1)

    x_, q_ = tf.split(actual, [3,4], axis=1)

    distance = tf.norm(tf.subtract(x_, x), axis=1)

    #arr = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
    #l_q = arr.unstack(q)
    #q_norms = [tf.norm(qwop) for qwop in l_q]
    #unit_q = l_q.stack(q / q_norms)

    norm_q = tf.norm(q, axis=1, keep_dims=True)

    unit_q = tf.divide(q, norm_q)

    angle = tf.abs(tf.subtract(_get_angle(q_), _get_angle(unit_q)))
    angle = tf.divide(tf.multiply(angle, 180.0), np.pi) # to degrees

    return tf.stack(
                (tf.reduce_mean(distance), tf.reduce_mean(angle)),
                axis=0
           )


def shared_dual_stream(model, lr=1e-3):

    net = model['PrePool']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    net = slim.conv2d_transpose(model['upper_input'],
                                tf.Dimension(int(net.get_shape()[3])),
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d_transpose(net,
                                tf.Dimension(int(net.get_shape()[3])/2),
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d_transpose(net,
                                tf.Dimension(int(net.get_shape()[3])/2),
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    left_1 = slim.conv2d(net,
                         32,
                         (3,3),
                         stride=1,
                         normalizer_fn=slim.batch_norm)

    right_1 = slim.conv2d(net,
                          32,
                          (3,3),
                          stride=1,
                          normalizer_fn=slim.batch_norm)

    left_and_share_right = tf.add(left_1, tf.multiply(right_1, 0.7))

    right_and_share_left = tf.add(right_1, tf.multiply(left_1, 0.7))

    loc_input = slim.flatten(left_and_share_right)

    ori_input = slim.flatten(right_and_share_left)

    loc_1 = slim.fully_connected(loc_input,
                                 1024,
                                 normalizer_fn=slim.batch_norm)

    ori_1 = slim.fully_connected(ori_input,
                                 1024,
                                 normalizer_fn=slim.batch_norm)

    loc_and_share_ori = tf.add(loc_1, tf.multiply(ori_1, 0.7))

    ori_and_share_loc = tf.add(ori_1, tf.multiply(loc_1, 0.7))

    loc = slim.fully_connected(loc_and_share_ori,
                               4,
                               normalizer_fn=slim.batch_norm)

    ori = slim.fully_connected(ori_and_share_loc,
                               5,
                               normalizer_fn=slim.batch_norm)

    logits = tf.concat((loc, ori), axis=1)

    loss = distance_with_learned_scale(
        predicted=logits,
        actual=model['labels']
    )

    distance = position_and_angle(
        predicted=logits,
        actual=model['labels']
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr
    ).minimize(loss)

    model['acc'] = distance


def decode_dual_stream(model, lr=1e-3):

    net = model['PrePool']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    net = slim.conv2d_transpose(model['upper_input'],
                                net.get_shape()[3],
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d_transpose(net,
                                tf.Dimension(int(net.get_shape()[3])),
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d_transpose(net,
                                tf.Dimension(int(net.get_shape()[3])/2),
                                (4,4),
                                stride=2,
                                normalizer_fn=slim.batch_norm)

    loc = slim.conv2d(net,
                      32,
                      (3,3),
                      stride=1,
                      normalizer_fn=slim.batch_norm)

    ori = slim.conv2d(net,
                      32,
                      (3,3),
                      stride=1,
                      normalizer_fn=slim.batch_norm)

    loc = slim.flatten(loc)

    ori = slim.flatten(ori)

    loc = slim.fully_connected(loc, 1024, normalizer_fn=slim.batch_norm)

    ori = slim.fully_connected(ori, 1024, normalizer_fn=slim.batch_norm)

    loc = slim.fully_connected(loc, 4, normalizer_fn=slim.batch_norm)

    ori = slim.fully_connected(ori, 5, normalizer_fn=slim.batch_norm)

    logits = tf.concat((loc, ori), axis=1)

    loss = distance_with_learned_scale(
        predicted=logits,
        actual=model['labels']
    )

    model['acc'] = position_and_angle(
        predicted=logits,
        actual=model['labels']
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr
    ).minimize(loss)

