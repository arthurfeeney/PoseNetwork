import tensorflow as tf
from custom_helper import *
slim = tf.contrib.slim

# The two cifar functions are
# used for simple testing other parts of the program.

def pose_upper(model, lr=1e-3, os=10):
    net = model['PrePool']

    model['upper_input'] = tf.placeholder(tf.float32, shape=net.get_shape())

    net = slim.avg_pool2d(model['upper_input'], net.get_shape()[1:3],
                          padding='VALID')

    net = slim.flatten(net)

    net = slim.fully_connected(net, 2048, activation_fn=None)

    loc = slim.fully_connected(net, 1024, activation_fn=None)

    ori = slim.fully_connected(net, 1024, activation_fn=None)

    loc = slim.fully_connected(loc, 3, activation_fn=None)

    ori = slim.fully_connected(ori, 4, activation_fn=None)

    logits = tf.concat((loc, ori), axis=-1)

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

def upper_net(model, lr, os):
    print('a work soon to be in progress :)')
