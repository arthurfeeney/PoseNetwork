def test(model, data, weight_file, verbose=False):
    print('test called')
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(weight_file)
        loader.restore(sess, tf.train.latest_checkpoint(
            '/data/zhanglab/afeeney/cifar_model/'
        ))

        images, labels = data.test()
        acc = 0
        count = 0

        batch_size = 16 # small default so it is a bit smoother.
        for index in range(len(labels)-batch_size-1):
            if verbose and index % (10 * batch_size) == 0:
                print('tested: %d images'%index)
            acc += model['acc'].eval(
                feed_dict = {
                    model['images']: images[index:index+batch_size],
                    model['labels']: labels[index:index+batch_size]
                }
            )
            count += 1
        test_acc = acc / count
        print('test acc: %g'%test_acc)
    return False

def mod_test(model, data, weight_file, verbose=False, batch_size=16):
    print('MODIFY_test called')
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(weight_file)
        loader.restore(sess, tf.train.latest_checkpoint(
            '/data/zhanglab/afeeney/cifar_model/'
        ))

        images, labels = data.test()
        acc = 0
        count = 0

        for index in range(len(labels)-batch_size-1):
            if verbose and index % (10 * batch_size) == 0:
                print('tested: %d images'%index)
            acc += model['acc'].eval(
                feed_dict = {
                    model['upper_input']: model['base_out'].eval(
                        feed_dict = {
                            model['images']: images[index:index+batch_size],
                            model['labels']: labels[index:index+batch_size]
                        }
                    ),
                    model['labels']: labels[index:index+batch_size]
                }
            )
            count += 1

        test_acc = acc / count
        print('test acc: %g'%test_acc)
    return False
