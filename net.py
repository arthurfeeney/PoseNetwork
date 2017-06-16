import tensorflow as tf
from custom_helper import *
slim = tf.contrib.slim
import math
import numpy as np

def euclidean_distance(predicted, actual, scale = 10):
    x, q = tf.split(tf.reduce_mean(predicted, axis=0), [3,4], axis=0)
    x_, q_ = tf.split(tf.reduce_mean(actual, axis=0), [3,4], axis=0)

    b = tf.constant(scale, dtype=tf.float32)

    return tf.norm(x_ - x) + \
           tf.multiply(b, tf.norm(q_ - tf.divide(q, tf.norm(q))))

def position_and_angle(predicted, actual):
    x, q = tf.split(tf.reduce_mean(predicted, axis=0), [3,4], axis=0)
    x_, q_ = tf.split(tf.reduce_mean(actual, axis=0), [3,4], axis=0)

    distance = tf.norm(x_ - x)

    angle = tf.norm(q_ - tf.divide(q, tf.norm(q)))

    return tf.stack((distance, angle), axis=0)

def decoder(model, lr=1e-3, os=7):

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

    distance = position_and_angle(
        logits,
        model['labels']
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr
    ).minimize(loss)

    correct = tf.equal(logits, model['labels'])

    #model['acc'] = loss

    model['acc'] = distance

def decode_dual_stream(model, lr=1e-3, os=7):

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

    loc = slim.conv2d(net, 32, (3,3), stride=1,
                      normalizer_fn=slim.batch_norm)

    ori = slim.conv2d(net, 32, (3,3), stride=1,
                      normalizer_fn=slim.batch_norm)

    loc = slim.flatten(loc)

    ori = slim.flatten(ori)

    loc = slim.fully_connected(loc, 3, normalizer_fn=slim.batch_norm)

    ori = slim.fully_connected(ori, 4, normalizer_fn=slim.batch_norm)

    logits = tf.concat((loc, ori), axis=1)

    loss = euclidean_distance(
        actual=model['labels'],
        predicted=logits
    )

    distance = position_and_angle(
        logits,
        model['labels']
    )

    model['step'] = tf.train.AdamOptimizer(
        learning_rate=lr
    ).minimize(loss)

    correct = tf.equal(logits, model['labels'])

    #model['acc'] = loss

    model['acc'] = distance

