def googlenet(lr, os):
    """
    my implementation of a modified googlenet. The 5x5 convolutions in the
    inception units were changed to a stack of 3x3 convolutions. It uese ELU
    activation for all of the conv. Batch normalization is used on every layer
    that has weights. ELU and BN are done inside the conv2d/dense function call
    in custom_helper. It uses Adam optimizer for gradient descent and
    softmax with logits for the loss function.
    """
    print('create_base_net called')
    # None means that the batch size can be interpreted.
    # GoogleNet takes 224x224 RGB images. The 3 is the RGB dimension!
    # lablels has a 1000 dimension because ImageNet has 1000 classes.

    #images = tf.placeholder(tf.float32, shape=[None,224,224,3])
    #labels = tf.placeholder(tf.float32, shape=[None,os])

    #
    # FOR CIFAR10 TEST. -> why its 32x32 not 224x224 at the moment.
    #

    images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    labels = tf.placeholder(tf.float32, shape=[None,os])
    #ximages = zeroPad(images, height=96, width=96)
    ximages = tf.image.resize_images(images, [224,224])
    #ximages = tf.image.resize_image_with_crop_or_pad()
    #tf.random_crop()

    conv1_7x7_s2 = conv2d(ximages, 64, shape=[7,7], stride=(2,2))

    # applies padding to height and width.
    conv1_zero_pad = zeroPad(conv1_7x7_s2, height=1, width=1)

    pool1_3x3_s2 = maxPool(conv1_zero_pad, size=[3,3], padding='VALID')

    conv2_3x3_reduce = conv2d(pool1_3x3_s2, numFilters=64, shape=[1,1])

    conv2_3x3_s1 = conv2d(conv2_3x3_reduce, numFilters=192, shape=[3,3])

    pool2_3x3_s2 = maxPool(conv2_3x3_s1, size=[3,3],padding='VALID')


    # Inception unit 3a.
    i_3a_output = inceptUnit(pool2_3x3_s2, [64,96,128,16,32,32,32])


    # Inception unit 3b.
    i_3b_output = inceptUnit(i_3a_output, [128,128,192,32,96,96,64])


    i_3b_output_zeroPad = zeroPad(i_3b_output)

    pool3_3x3_s2 = maxPool(i_3b_output_zeroPad,
                           size=[3,3],padding='VALID')


    #inception unit 4a.
    i_4a_output = inceptUnit(pool3_3x3_s2, [192,96,208,16,48,48,64])


    #inception unit 4b.
    i_4b_output = inceptUnit(i_4a_output, [160,112,224,24,64,64,64])

    #inception unit 4c.
    i_4c_output = inceptUnit(i_4b_output, [128,128,256,24,64,64,64])


    # inception unit 4d.
    i_4d_output = inceptUnit(i_4c_output, [112,144,228,32,64,64,64])


    # inception unit 4e.
    i_4e_output = inceptUnit(i_4d_output, [256,160,320,32,128,128,128])


    i_4e_output_zeroPad = zeroPad(i_4e_output)

    pool4_3x3_s2 = maxPool(i_4e_output_zeroPad,
                           size=[3,3], stride=[2,2], padding='VALID')


    #inception unit 5a.
    i_5a_output = inceptUnit(pool4_3x3_s2, [256,160,320,32,128,128,128])


    # inception unit 5b
    i_5b_output = inceptUnit(i_5a_output, [384,192,384,48,128,128,128])


    avg_pool = tf.nn.avg_pool(i_5b_output,
                              ksize=[1,7,7,1],
                              strides=[1,1,1,1],
                              padding='VALID',
                              name='base_output1')

    avg_pool_flat = tf.reshape(avg_pool, [-1, 1024])

    dropout_layer = tf.nn.dropout(avg_pool_flat, .4) # not needed with BN?

    # 1000 is the output size
    network_output = dense(dropout_layer, 10)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=network_output
           )

    loss = tf.reduce_mean(cross_entropy)

    step = tf.train.AdamOptimizer(
                learning_rate = lr,
                epsilon = 1 # find
           ).minimize(loss)

    correct = tf.equal(tf.argmax(network_output, 1), tf.argmax(labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # also possibly return the last inception unit as a way to make mods.
    return images, labels, step, accuracy #, i_5b_output
