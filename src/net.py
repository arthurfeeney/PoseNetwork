import tensorflow as tf
import numpy as np

layers = tf.contrib.layers

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

    loss_pos = (tf.norm(x_ - x, ord=1, axis=1) * tf.exp(-s_x)) + s_x

    norm_q = tf.norm(q, axis=1, keep_dims=True)

    unit_q = tf.divide(q, norm_q)

    loss_ori = (tf.norm(q_ - unit_q, ord=1, axis=1) * tf.exp(-s_q)) + s_q

    return loss_pos + loss_ori

def _position_and_angle(predicted, actual):
    """
    computes the position and angle error for printing and test accuracy.
    it uses axis 1 to apply function to each element of batch.
    The math functions should all be element wise already.
    """
    x, q = tf.split(predicted, [3,4], axis=1)

    x_, q_ = tf.split(actual, [3,4], axis=1)

    distance = tf.norm(x_ - x, axis=1)

    norm_q = tf.norm(q, axis=1, keep_dims=True)

    unit_q = tf.divide(q, norm_q)

    radian_angle = tf.abs(tf.norm(q_ - unit_q, axis=1))

    angle = tf.divide(tf.multiply(radian_angle, 180.0), np.pi)

    return tf.stack(
                (tf.reduce_mean(distance), tf.reduce_mean(angle)),
                axis=0
           )

def _stream(input, scope_name, reuse, is_training):
    stream = layers.conv2d(input, 32, [3,3], stride=1, padding='SAME')
    stream = layers.batch_norm(stream, decay=.95, updates_collections=None,
                               reuse=reuse, scope=scope_name+str(1),
                               is_training=is_training)
    stream = tf.nn.relu(stream)
    stream = layers.conv2d(stream, 32, [3,3], stride=1, padding='SAME')
    stream = layers.batch_norm(stream, decay=.95, updates_collections=None,
                               reuse=reuse, scope=scope_name+str(2),
                               is_training=is_training)
    stream = tf.nn.relu(stream)
    stream = layers.flatten(stream)
    stream = layers.dense(stream, 1024)
    output = layers.batch_norm(stream, decay=.95, updates_collections=None,
                               reuse=reuse, scope=scope_name+str(1),
                               is_training=is_training)
    tf.nn.relu(output)
    return output

def shared_dual_stream(model, lr=1e-4, reuse=False):
    """
    My own network design. Attempt to separate the regression of position
    and orientation into separate streams while still sharing the information
    of what the other stream is doing.
    """
    net = model['PrePool']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    model['keep_prob'] = tf.placeholder(tf.float32)

    model['is_training'] = tf.placeholder(tf.bool)

    with tf.variable_scope('model', reuse=reuse):

        net = layers.conv2d_transpose(model['upper_input'], 768, [4,4],
                                      stride=2)
        net = layers.batch_norm(net, decay=0.95, updates_collections=None,
                                reuse=reuse, scope='b1',
                                is_training=model['is_training'])
        net = tf.nn.relu(net)

        net = layers.conv2d_transpose(model['upper_input'], 384, [4,4],
                                      stride=2)
        net = layers.batch_norm(net, decay=0.95, updates_collections=None,
                                reuse=reuse, scope='b2',
                                is_training=model['is_training'])
        net = tf.nn.relu(net)

        left = _stream(net, scope_name='left', reuse=reuse,
                       is_training=model['is_training'])

        right = _stream(net, scope_name='right', reuse=reuse,
                        is_training=model['is_training'])

        lsr = tf.Variable(.5, dtype=tf.float32, trainable=True)
        qsr = tf.Variable(.5, dtype=tf.float32, trainable=True)

        net = ((left / (2*lsr)) + lsr) + ((right / (2*qsr)) + qsr)

        net = layers.dropout(net, keep_prob=model['keep_prob'],
                             is_training=model['is_training'])

        loc = layers.dense(net, 3)

        ori = layers.dense(net, 4)

        logits = tf.concat((loc, ori), axis=1)

        s_x = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        s_q = tf.Variable(-3.0, dtype=tf.float32, trainable=True)

        loss = _distance_with_learned_scale(predicted=logits,
                                            actual=model['labels'], s_x=s_x,
                                            s_q=s_q)

        model['acc'] = _position_and_angle(predicted=logits,
                                           actual=model['labels'])

    update = 'qwop'
    if not reuse:
        update = tf.train.AdamOptimizer(
            learning_rate=lr,
            epsilon=1e-6
        ).minimize(loss)
    return update

def decode_dual_stream(model, lr=1e-4, reuse=False):
    """
    Implementation of network from the hourglass networks paper.
    """
    net = model['Conv2d_7b_1x1']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    model['keep_prob'] = tf.placeholder(tf.float32)

    model['is_training'] = tf.placeholder(tf.bool)

    with tf.variable_scope('model', reuse=reuse):

        net = layers.conv2d_transpose(model['upper_input'], net.get_shape()[3],
                                    (4,4), stride=2)
        net = layers.batch_norm(net, decay=0.95, updates_collections=None,
                                reuse=reuse, scope='b1',
                                is_training=model['is_training'])
        net = tf.nn.relu(net)

        net = layers.conv2d_transpose(net, 768, (4,4), stride=2)

        net = layers.batch_norm(net, decay=0.95, updates_collections=None,
                                reuse=reuse, scope='b2',
                                is_training=model['is_training'])
        net = tf.nn.relu(net)

        net = layers.conv2d_transpose(net, 384, (4,4), stride=2)
        net = layers.batch_norm(net, decay=0.95, updates_collections=None,
                                reuse=reuse, scope='b3',
                                is_training=model['is_training'])
        net = tf.nn.relu(net)

        net = layers.conv2d(net, 32, (3,3), stride=1, padding='SAME')
        net = layers.batch_norm(net, decay=0.95, updates_collections=None,
                                reuse=reuse, scope='b4',
                                is_training=model['is_training'])
        net = tf.nn.relu(net)

        net = layers.flatten(net)

        net = tf.layers.dense(net, 1024)
        net = layers.batch_norm(net, decay=0.95, updates_collections=None,
                                reuse=reuse, scope='b5',
                                is_training=model['is_training'])
        net = tf.nn.relu(net)

        net = layers.dropout(net, keep_prob=model['keep_prob'],
                             is_training=model['is_training'])

        loc = tf.layers.dense(net, 3)

        ori = tf.layers.dense(net, 4)

        logits = tf.concat((loc, ori), axis=1)

        s_x = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        s_q = tf.Variable(-3.0, dtype=tf.float32, trainable=True)

        loss = _distance_with_learned_scale(predicted=logits,
                                        actual=model['labels'],
                                        s_x=s_x, s_q=s_q)

        model['acc'] = _position_and_angle(predicted=logits,
                                       actual=model['labels'])

    update = 'qwop'
    if not reuse:
        update = tf.train.AdamOptimizer(
            learning_rate=lr,
            epsilon=1e-6
        ).minimize(loss)
    return update

