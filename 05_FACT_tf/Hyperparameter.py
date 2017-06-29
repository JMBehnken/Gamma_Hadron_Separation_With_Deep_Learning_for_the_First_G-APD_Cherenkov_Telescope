# Import of every needed library
import pickle
import gzip
import numpy as np
import tensorflow as tf
import os
import random
from multiprocessing import Pool
import csv
import time
#import myTelegram

path = '/fhgfs/users/jbehnken/randomized_Conv_Data' # Path to preprocessed data
num_files = 1000

# Load pickled data and split it into pictures and labels
def load_data(file):
    with gzip.open(path+'/'+file, 'rb') as f:
        data_dict = pickle.load(f)
    pic = data_dict['Image']
    lab = data_dict['Label']
    return (pic, lab)

# Randomizing the files to load
loading_files = os.listdir(path)
np.random.shuffle(loading_files)

# Pool-load pickled data and split it into pictures and labels (list)
p = Pool()
data = p.map(load_data, loading_files[:num_files])
pics, labs = zip(*data)
del data, p

# Concatenate the data to a single np.array
pic = np.concatenate(pics)
lab = np.concatenate(labs)
del pics, labs

# Randomize and split the data into train/validation/test dataset
p = np.random.permutation(len(pic))

valid_dataset = pic[p][:50000]
valid_labels = lab[p][:50000]
#test_dataset = pic[p][50000:50000]
#test_labels = lab[p][50000:50000]
train_dataset = pic[p][50000:]
train_labels = lab[p][50000:]
del p, pic, lab

# Hyperparameter for the model (fit manually)
num_labels = 2 # gamma or proton
num_channels = 1 # it is a greyscale image

# Quantity of batches to train with
for num_steps in [1000001]:
    
    # Different learning rates for the optimizer
    for learning_rate in [ 0.001]:
        
        # Quantity of the events in one batch
        for batch_size in [128]:
            
            # Size of the conv 2d filter [ixi]
            for patch_size in [5]:
                
                # Quantity of the filters of the first conv2d-layer
                for depth in  [8]:
                    
                    # Quantity of the nodes in the first hidden layer
                    for num_hidden in  [16]:

                        # Path to logfiles and correct file name
                        start = time.time()
                        LOGDIR = '/fhgfs/users/jbehnken/tf_logs/small_logs'
                        logcount = str(len(os.listdir(LOGDIR)))
                        hparams = 'Adam_lr={}_bs={}_ps={}_d={}_nh={}_ns={}'.format(learning_rate, batch_size, patch_size, depth, num_hidden, num_steps)

                        
                        # Build the graph
                        gpu_config = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
                        session_conf = tf.ConfigProto(gpu_options=gpu_config, intra_op_parallelism_threads=18, inter_op_parallelism_threads=18)
                        tf.reset_default_graph()
                        sess = tf.Session(config=session_conf)

                        
                        # Create tf.variables for the three different datasets
                        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels), name='training_data')
                        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='training_labels')
                        
                        #tf.summary.image('input', tf_train_dataset, 6)
                        
                        tf_valid_dataset = tf.constant(valid_dataset, name='validation_data')
                        tf_valid_labels = tf.constant(valid_labels, name='validation_labels')
                        #tf_test_dataset = tf.constant(test_dataset, name='testing_data')
                        #tf_test_labels = tf.constant(test_labels, name='testing_labels')


                        # First layer is a convolution layer
                        with tf.name_scope('conv2d_1'):
                            layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name='W_1')
                            layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth]), name='B_1')

                            conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
                            hidden = tf.nn.relu(conv + layer1_biases)
                            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                            #tf.summary.histogram("weights", layer1_weights)
                            #tf.summary.histogram("biases", layer1_biases)
                            #tf.summary.histogram("activations", hidden)


                        # Second layer is a convolution layer
                        with tf.name_scope('conv2d_2'):
                            layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, 2*depth], stddev=0.1), name='W_2')
                            layer2_biases = tf.Variable(tf.constant(1.0, shape=[2*depth]), name='B_2')

                            conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME') 
                            hidden = tf.nn.relu(conv + layer2_biases)
                            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                            #tf.summary.histogram("weights", layer2_weights)
                            #tf.summary.histogram("biases", layer2_biases)
                            #tf.summary.histogram("activations", hidden)


                        # The reshape produces an input vector for the dense layer
                        with tf.name_scope('reshape'):
                            shape = pool.get_shape().as_list()
                            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])


                        # Third layer is a dense layer
                        with tf.name_scope('fc_1'):
                            layer3_weights = tf.Variable(tf.truncated_normal([12*12*2*depth, num_hidden], stddev=0.1), name='W_3')
                            layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B_3')

                            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

                            #tf.summary.histogram("weights", layer3_weights)
                            #tf.summary.histogram("biases", layer3_biases)
                            #tf.summary.histogram("activations", hidden)


                        # Fourth layer is a dense output layer
                        with tf.name_scope('fc_2'):
                            layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), name='W_4')
                            layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B_4')

                            output = tf.matmul(hidden, layer4_weights) + layer4_biases

                            #tf.summary.histogram("weights", layer4_weights)
                            #tf.summary.histogram("biases", layer4_biases)
                            #tf.summary.histogram("activations", output)


                        # Computing the loss of the model
                        with tf.name_scope('loss'):
                            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')
                            tf.summary.scalar("loss", loss)


                        # Optimizing the model
                        with tf.name_scope('optimizer'):
                            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


                        # Predictions for the training, validation, and test data
                        with tf.name_scope('prediction'):
                            train_prediction = tf.nn.softmax(output)


                        with tf.name_scope('accuracy'):
                            correct_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels, 1))
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                            tf.summary.scalar('batch_accuracy', accuracy)
                            
                            
                        with tf.name_scope('validation'):
                            pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_valid_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                            pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME')  + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                            shape = pool_2.get_shape().as_list()
                            reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
                            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
                            valid_prediction = tf.nn.softmax(tf.matmul(hidden, layer4_weights) + layer4_biases)
                            
                            correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(valid_labels, 1))
                            valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                            tf.summary.scalar('validation_accuracy', valid_accuracy)
                            
                            
                        with tf.name_scope('auc'):
                            auc = tf.metrics.auc(labels=tf_valid_labels, predictions=valid_prediction, curve='ROC')
                            tf.summary.scalar('validation_auc_0', auc[0])
                            #tf.summary.scalar('validation_auc_1', auc[1])


                        summ = tf.summary.merge_all()
                        saver = tf.train.Saver()

                        sess.run(tf.global_variables_initializer())
                        sess.run(tf.local_variables_initializer())
                        writer = tf.summary.FileWriter(LOGDIR+'/'+logcount+hparams)
                        writer.add_graph(sess.graph)


                        # Iterating over num_setps 
                        for step in range(num_steps):
                            # Computing the offset to move over the training dataset
                            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                            # Getting the batchdata
                            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
                            # Getting the batchlabels
                            batch_labels = train_labels[offset:(offset + batch_size), :]
                            # Creating a feed_dict to train the model on in this step
                            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                            # Train the model for this step
                            _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

                            # Updating the output to stay in touch with the training process
                            if (step % 1000 == 0):
                                [acc, val, auc_val, s] = sess.run([accuracy, valid_accuracy, auc, summ], feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels})
                                #print('Minibatch loss at step %d: %f' % (step, l))
                                #print('Minibatch accuracy: %.1f%%' % (acc*100))
                                #print('Validation accuracy: %.1f%%' % (val*100))
                                #print('Auc: %.2f, %.2f' % (auc_val[0], auc_val[1]))
                                #writer.add_summary(s, step)
                                
                                if step == 0:
                                    stopping_auc = 0.0
                                    sink_count = 0
                                else:
                                    if auc_val[0] > stopping_auc:
                                        stopping_auc = auc_val[0]
                                        sink_count = 0
                                        saver.save(sess, 'CNN_Test_Model_3')
                                    else:
                                        sink_count += 1
                                print('St_auc: {}, sc: {},val: {}, Step: {}'.format(stopping_auc, sink_count, val*100, step))
                                if sink_count == 5:
                                    break   
                        
                        #saver.save(sess, 'CNN_Test_Model_2')

                        dauer = time.time() - start
                        with open('/home/jbehnken/05_FACT_tf/Hyperparameter.csv', 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([learning_rate, batch_size, patch_size, depth, num_hidden, val*100, stopping_auc, step, dauer])

#message = "I've finished calculating for you."
#myTelegram.sendTelegram(message)