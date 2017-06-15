import tensorflow as tf
from custom_helper import *
slim = tf.contrib.slim
import math
import numpy as np
# The two cifar functions are
# used for simple testing other parts of the program.

def euclidean_distance(predicted, actual, scale = 10):
    x, q = tf.split(tf.reduce_mean(predicted, axis=0), [3,4], axis=0)
    x_, q_ = tf.split(tf.reduce_mean(actual, axis=0), [3,4], axis=0)

    b = tf.constant(scale, dtype=tf.float32)

    return tf.norm(x_ - x) + \
           tf.multiply(b, tf.norm(q_ - tf.divide(q, tf.norm(q))))

def pose_upper(model, lr=1e-3, os=7):

    net = model['PrePool']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    net = slim.conv2d_transpose(model['upper_input'], net.get_shape()[3],
                                (4,4), stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d_transpose(net, net.get_shape()[3], (4,4), stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d_transpose(net, net.get_shape()[3], (4,4), stride=2,
                                normalizer_fn=slim.batch_norm)

    net = slim.conv2d(net, 32, (3,3), stride=1,
                      normalizer_fn=slim.batch_norm)

    net = slim.flatten(net)

    net = slim.fully_connected(net, 2048, normalizer_fn=slim.batch_norm)

    loc = slim.fully_connected(net, 3, normalizer_fn=slim.batch_norm)

    ori = slim.fully_connected(net, 4, normalizer_fn=slim.batch_norm)

    logits = tf.concat((loc, ori), axis=1)

    loss = euclidean_distance(
        actual=model['labels'],
        predicted=logits
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr
    ).minimize(loss)

    correct = tf.equal(logits, model['labels'])

    model['acc'] = loss


def incept_upper_mod(model, lr=1e-3, os=10):
    net = model['PrePool']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    net = slim.avg_pool2d(model['upper_input'], net.get_shape()[1:3],
                        padding='VALID')

    net = slim.flatten(net)

    net = slim.dropout(net, .5, is_training=True)

    model['PreLogitsFlatten'] = net
    logits = slim.fully_connected(net, os, activation_fn=None)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=model['labels'],
        logits=logits
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr
    ).minimize(loss)

    correct = tf.equal(tf.argmax(logits, 1),
                       tf.argmax(model['labels'], 1))

    model['acc'] = tf.reduce_mean(tf.cast(correct, tf.float32))

def cifar_upper_net(model, lr=1e-3, os=10):
    print('CIFAR_upper_net')

    model['upper_input'] = tf.placeholder(tf.float32, shape=[None,8,8,64])

    upper_input_flat = tf.reshape(model['upper_input'], [-1, 4096])

    fc1 = dense(upper_input_flat, size=1024, name='poop')

    mod_out = dense(fc1, size=10, name='ultrapoo')

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=model['labels'],
        logits=mod_out
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr,
        name='Adam_loss'
    ).minimize(loss)

    correct = tf.equal(tf.argmax(mod_out, 1),
                       tf.argmax(model['labels'], 1),
                       name='shit')

    model['acc'] = tf.reduce_mean(tf.cast(correct, tf.float32), name='yum')

    return model
