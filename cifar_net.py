def cifar_base_net(lr=1e-3, os=10):
    print('CIFAR_base_net')
    images = tf.placeholder(tf.float32, shape=[None,32,32,3], name='lower_img')
    labels = tf.placeholder(tf.float32, shape=[None, os], name='lower_label')
    conv1_3x3_s1 = conv2d(images, 32, shape=[3,3], stride=(1,1), name='conv1')
    pool1_2x2_s2 = maxPool(conv1_3x3_s1, name='pool1')
    conv2_3x3_s1 = conv2d(pool1_2x2_s2, 64, shape=[3,3], stride=(1,1),
                          name='conv2')
    pool2_2x2_s2 = maxPool(conv2_3x3_s1, name='base_output1')
    pool2_flat = tf.reshape(pool2_2x2_s2, [-1, 4096], name='pool_flat')



    fc1_left = dense(pool2_flat, 1024, name='dense_left')
    fc1_right = dense(pool2_flat, 1024, name='dense_right')
    fc1_combine = tf.concat((fc1_left,
                          fc1_right),
                          axis=-1, name='combine')
    output = dense(fc1_combine, 10, name='lower_output')
    loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=output,
                name='lower_loss'
            )
    step = tf.train.AdamOptimizer(
                learning_rate=lr,
                name='lower_Adam'
            ).minimize(loss)
    correct = tf.equal(tf.argmax(output, 1),
                       tf.argmax(labels, 1),
                       name='lower_correct')
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='lower_acc')
    return images, labels, step, accuracy
