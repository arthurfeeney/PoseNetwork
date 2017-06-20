def train(model, data, save_weight_file, init_weight_file=None,
          batch_size=32, epoch=1, verbose=False):
    with tf.Session() as sess:
        if init_weight_file is not None:
            loader = tf.train.import_meta_graph(init_weight_file)
            loader.restore(sess, tf.train.latest_checkpoint(
                '/data/zhanglab/afeeney/cifar_model/'
            ))
        else:
            sess.run(tf.global_variables_initializer())

        for e in range(epoch):
            data.shuffleData()
            # modifying number of steps.
            for epi in range(0, data.trainSize()-batch_size-1, batch_size):
                ib, lb = data.trainBatch(batch_size)
                if verbose and epi % (20*batch_size) == 0:
                    # this should use validation set for ImageNet
                    # keeping score of the last validation set.
                    # also after the training part.
                    batchAccuracy = model['acc'].eval(
                        feed_dict = {
                            model['images']: ib,
                            model['labels']: lb
                        }
                    )
                    print('step %d, acc %g'%(epi, batchAccuracy))
                model['step'].run(
                    feed_dict = {
                        model['images']: ib,
                        model['labels']: lb
                    }
                )
            saver = tf.train.Saver()
            saver.save(sess, save_weight_file, global_step=e)

def mod_train(model, data, save_weight_file,
              init_weight_file, batch_size=32, epoch=1, verbose=False):
    with tf.Session() as sess:

        model = cifar_upper_net(model)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        loader = tf.train.import_meta_graph(init_weight_file)
        loader.restore(sess, tf.train.latest_checkpoint(
            '/data/zhanglab/afeeney/cifar_model/'
        ))

        model['base_out'] = \
            tf.get_default_graph().get_tensor_by_name('base_output1:0')

        for e in range(epoch):
            data.shuffleData()
            for epi in range(0, data.trainSize()-batch_size-1, batch_size):
                batch = data.trainBatch(batch_size)
                model['step'].run(
                    feed_dict = {
                        model['upper_input']: model['base_out'].eval(
                            feed_dict = {
                                model['images']: batch[0],
                                model['labels']: batch[1]
                            }
                        ),
                        model['labels']: batch[1]
                    }
                )
        saver.save(sess, save_weight_file, global_step=0)
    return model
