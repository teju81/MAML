import tensorflow as tf
from tensorflow.python.platform import flags
from datagenerator import datagenerator as dg
from model import maml

flags = tf.app.flags 
FLAGS = tf.app.flags.FLAGS


## General options
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_bool('data_resize', False, 'Set to True if you want to generate the resized omniglot data again')
flags.DEFINE_bool('data_generation', False, 'Set to True if you want to generate the data again')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.1, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 0.1, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 5, 'number of inner gradient updates during training.')
flags.DEFINE_string('training_data_dir', 'data/training/', 'Directory for the training data')
flags.DEFINE_string('testing_data_dir', 'data/testing/', 'Directory for the testing data')
flags.DEFINE_string('resized_data_dir', 'data/resized_data/', 'Directory for the resized omniglot data')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_string('reg', 'dropout', 'dropout, weight_decay or None')
flags.DEFINE_float('dropout_rate', 0.15, 'dropout rate')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', True, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'data/tmp', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')

flags.DEFINE_integer('summary_interval', 10, 'Number of Iterations between two summaries')
flags.DEFINE_integer('save_interval', 50, 'Number of Iterations between two saved checkpoints')
flags.DEFINE_integer('print_interval', 10, 'Number of Iterations between two prints')

flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def main():
    
    test_num_updates = 10
    data_generator = dg.DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory


    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input

    tf_data_load = True

    if FLAGS.train: # only construct training model if needed
        image_tensor = data_generator.train_inputs['images']
        label_tensor = data_generator.train_inputs['labels']
        iter_init_op = data_generator.train_inputs['iter_init_op']

        inputa = tf.slice(image_tensor, [0,0,0], [-1,FLAGS.update_batch_size,-1])
        inputb = tf.slice(image_tensor, [0,FLAGS.update_batch_size,0], [-1, -1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,FLAGS.update_batch_size,-1])
        labelb = tf.slice(label_tensor, [0,FLAGS.update_batch_size,0], [-1,-1,-1])
        train_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb, 'iter_init_op': iter_init_op}
    
    image_tensor = data_generator.metaval_inputs['images']
    label_tensor = data_generator.metaval_inputs['labels']
    iter_init_op = data_generator.metaval_inputs['iter_init_op']
    inputa = tf.slice(image_tensor, [0,0,0], [-1,FLAGS.update_batch_size,-1])
    inputb = tf.slice(image_tensor, [0,FLAGS.update_batch_size,0], [-1, -1,-1])
    labela = tf.slice(label_tensor, [0,0,0], [-1,FLAGS.update_batch_size,-1])
    labelb = tf.slice(label_tensor, [0,FLAGS.update_batch_size,0], [-1,-1,-1])
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb, 'iter_init_op': iter_init_op}

    model = maml.MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=train_input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=2)

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')


    train(model, saver, exp_string, data_generator)

if __name__ == '__main__':
    main()
