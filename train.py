import tensorflow as tf
from tensorflow.python.platform import flags

flags = tf.app.flags 
FLAGS = tf.app.flags.FLAGS

def train(model, saver, exp_string, data_generator):
    SUMMARY_INTERVAL = FLAGS.summary_interval
    SAVE_INTERVAL = FLAGS.save_interval
    PRINT_INTERVAL = FLAGS.print_interval
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(model.iter_init_op)
    
    if FLAGS.log:
      train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    prelosses, postlosses = [], []

    multitask_weights, reg_weights = [], []
    
    resume_itr = 0
    model_file = None

    if FLAGS.resume or not FLAGS.train:
      model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
      if model_file:
          ind1 = model_file.index('model')
          resume_itr = int(model_file[ind1+5:])
          print("Restoring model weights from " + model_file)
          saver.restore(sess, model_file)
    
    
    print('Done initializing, starting training.')
    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        # Select one of the two optimization ops depending on if it is pre training or meta training iteration
        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])

        try:
          result = sess.run(input_tensors, feed_dict)
        except tf.errors.OutOfRangeError:
          print("End of Dataset Reached... Starting over again!!!!")
          sess.run(model.iter_init_op)
          result = sess.run(input_tensors, feed_dict)



        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            feed_dict = {}
            input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))
    return
