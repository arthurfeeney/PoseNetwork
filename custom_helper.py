import tensorflow as tf


global batch_count
global bias_count
def set_batch_count():
    global batch_count
    global bias_count
    batch_count = 0
    bias_count = 0

# constructs a convolutional of the arguments. Just makes implement a bit easier
# stride defaults to 1 and padding defaults to SAME
def conv2d(convInput,
           numFilters,
           shape=[3,3],
           stride=(1,1),
           padding='SAME',
           name=None):
    global batch_count
    batch_count += 1
    return tf.nn.elu(_batchNormalize(tf.layers.conv2d(
                inputs = convInput,
                filters = numFilters, # the number of filters/output size
                kernel_size = shape,  # the size of the conv window
                strides = stride,
                padding = padding,
                activation = None, # None here allows me to do it after BN.
                kernel_regularizer = None,
                name=name), name=('batch'+str(batch_count))
            ))

# constructs a pooling layer of the arguments.
def maxPool(poolInput, size=[2,2], stride=[2,2], padding='SAME', name=None):
    return tf.nn.max_pool(
                poolInput,
                ksize = [1] + size + [1],
                strides = [1] + stride + [1],
                padding = padding,
                name=name
            )

def zeroPad(padInput, height=1, width=1, mode='CONSTANT'):
    return tf.pad(padInput,
                  paddings=[[0,0],[height,height],[width,width],[0,0]],
                  mode=mode)


# function to make inception unit.
def inceptUnit(inceptInput, filters):
    i_1x1 = conv2d(inceptInput, filters[0], shape=[1,1])

    i_3x3_reduce = conv2d(inceptInput, filters[1], shape=[1,1])

    i_3x3 = conv2d(i_3x3_reduce, filters[2], shape=[3,3])

    i_5x5_reduce = conv2d(inceptInput, filters[3], shape=[1,1])

    i_5x5_1 = conv2d(i_5x5_reduce, filters[4], shape=[3,3])

    i_5x5_2 = conv2d(i_5x5_1, filters[5], shape=[3,3])

    pool = maxPool(inceptInput, size=[3,3], stride=[1,1])

    pool_proj = conv2d(pool, filters[6], shape=[1,1])

    return tf.concat((i_1x1, i_3x3, i_5x5_2, pool_proj), axis=-1)

# returns a batch normalization layer.
def _batchNormalize(normInput, name=None):
    global batch_count
    batch_count += 1
    return tf.layers.batch_normalization(
            inputs=normInput,
            name=name)

def dense(denseInput, size, name=None):
    global batch_count
    batch_count += 1
    return _batchNormalize(
                tf.layers.dense(
                    inputs=denseInput,
                    units=size,
                    name=name,
                    use_bias=False
                ),
                name=('batch'+str(batch_count))
            )

# constructs a bias variable.
def biasVariable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

# returns random weights. Probably not needed and bad initialization method.
# however, batch normalization makes it less important...
def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)
