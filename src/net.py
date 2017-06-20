import tensorflow as tf

slim = tf.contrib.slim

def euclidean_distance(predicted, actual, scale = 10):
    x, q = tf.split(tf.reduce_mean(predicted, axis=0), [3,4], axis=0)
    x_, q_ = tf.split(tf.reduce_mean(actual, axis=0), [3,4], axis=0)

    b = tf.constant(scale, dtype=tf.float32)

    return tf.norm(x_ - x) + \
           tf.multiply(b, tf.norm(q_ - tf.divide(q, tf.norm(q))))


# from geometric loss functions for cpr with dl
def distance_with_learned_scale(predicted, actual):
    position, orientation = tf.split(tf.reduce_mean(predicted, axis=0),
                              [4,5],
                              axis=0)

    x, s_x = tf.split(position, [3,1], axis=0)

    q, s_q = tf.split(orientation, [4, 1], axis=0)

    x_, q_ = tf.split(tf.reduce_mean(actual, axis=0), [3,4], axis=0)

    left = tf.add(tf.multiply(tf.norm(x_ - x), tf.exp(-s_x)), s_x)

    unit_q = tf.divide(q, tf.norm(q))

    right = tf.add(tf.multiply(tf.norm(q_ - unit_q), tf.exp(s_q)), s_q)

    return left + right


def position_and_angle(predicted, actual):
    position, orientation = \
        tf.split(tf.reduce_mean(predicted, axis=0), [4,5], axis=0)

    x, _ = tf.split(position, [3,1], axis=0)

    q, _ = tf.split(orientation, [4,1], axis=0)

    x_, q_ = tf.split(tf.reduce_mean(actual, axis=0), [3,4], axis=0)

    distance = tf.norm(x_ - x)

    unit_q = tf.divide(q, tf.norm(q))

    cos_angle = tf.reduce_sum(tf.multiply(unit_q, q))

    angle = tf.acos(tf.clip_by_value(cos_angle, -1.0, 1.0))

    #angle = tf.norm(q_ - unit_q)

    return tf.stack((distance, angle), axis=0)

def decode_dual_stream(model, lr=1e-3):

    net = model['PrePool']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    net = slim.conv2d_transpose(model['upper_input'],
                                net.get_shape()[3],
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

    loc = slim.conv2d(net, 32, (3,3), stride=1,
                      normalizer_fn=slim.batch_norm)

    ori = slim.conv2d(net, 32, (3,3), stride=1,
                      normalizer_fn=slim.batch_norm)

    loc = slim.flatten(loc)

    ori = slim.flatten(ori)

    loc = slim.fully_connected(loc, 4, normalizer_fn=slim.batch_norm)

    ori = slim.fully_connected(ori, 5, normalizer_fn=slim.batch_norm)

    logits = tf.concat((loc, ori), axis=1)

    loss = distance_with_learned_scale(
        predicted=logits,
        actual=model['labels']
    )

    distance = position_and_angle(
        logits,
        model['labels']
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr
    ).minimize(loss)

    model['acc'] = distance

