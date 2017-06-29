# Import of every needed library
from multiprocessing import Pool
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle
import gzip
import time
import csv
import os

mc_data_path = '/fhgfs/users/jbehnken/01_Data/01_MC_Data' # Path to preprocessed data
num_files = 500 # Number of files to load - 1 file = 1000 events
events_in_validation = 10000
number_of_nets = 4

save_model_path = '/fhgfs/users/jbehnken/01_Data/04_Models'
model_name = 'cfff'
title_name = 'Random_Plotting'

file_paths = os.listdir(save_model_path)
for path in file_paths:
    name = '_' + model_name
    if path.endswith(name):
        correct_path = path 

if 'correct_path' in locals():
    folder_path = os.path.join(save_model_path, correct_path)
else:
    folder_number = len(os.listdir(save_model_path))+1
    folder_path = save_model_path + '/' + str(folder_number) + '_' + model_name
    os.mkdir(folder_path)
    
    with open(os.path.join(folder_path, model_name+'_Hyperparameter.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['Learning_Rate','Batch_Size','Patch_Size','Depth','Hidden_Nodes','Accuracy','Auc','Steps', 'Early_Stopped','Time', 'Title'])

# Load pickled data and split it into pictures and labels
def load_data(file):
    with gzip.open(mc_data_path+'/'+file, 'rb') as f:
        data_dict = pickle.load(f)
    pic = data_dict['Image']
    lab = data_dict['Label']
    return (pic, lab)

# Randomizing the files to load
loading_files = os.listdir(mc_data_path)
np.random.shuffle(loading_files)

# Pool-load pickled data and split it into pictures and labels (list)
p = Pool()
data = p.map(load_data, loading_files[:num_files])
pics, labs = zip(*data)

# Concatenate the data to a single np.array
pic = np.concatenate(pics)
lab = np.concatenate(labs)
del pics, labs

# Randomize and split the data into train/validation/test dataset
p = np.random.permutation(len(pic))

valid_dataset = pic[p][:events_in_validation]
valid_labels = lab[p][:events_in_validation]
train_dataset = pic[p][events_in_validation:]
train_labels = lab[p][events_in_validation:]
del p, pic, lab

# Hyperparameter for the model (fit manually)
num_labels = 2 # gamma or proton
num_channels = 1 # it is a greyscale image

num_steps = [50001] * number_of_nets
learning_rate = [0.001] * number_of_nets # 0.001
batch_size = np.random.randint(64, 257, size=number_of_nets) # 64 - 257
patch_size = np.random.randint(0, 2, size=number_of_nets)*2+3 # 3 / 5
depth = np.random.randint(8, 33, size=number_of_nets) # 8 - 33
num_hidden = np.random.randint(8, 257, size=number_of_nets) # 8 - 257

hyperparameter = zip(num_steps, learning_rate, batch_size, patch_size, depth, num_hidden)

df = pd.read_csv(os.path.join(folder_path, model_name+'_Hyperparameter.csv'))
if len(df['Auc']) > 0:
    best_auc = df['Auc'].max()
else:
    best_auc = 0

for num_steps, learning_rate, batch_size, patch_size, depth, num_hidden in hyperparameter:
    try:
        # Path to logfiles and correct file name
        start = time.time()
        LOGDIR = '/fhgfs/users/jbehnken/tf_logs/small_logs'
        logcount = str(len(os.listdir(LOGDIR)))
        hparams = 'bs={}_ps={}_d={}_nh={}_ns={}'.format(batch_size, patch_size, depth, num_hidden, num_steps)
    
        # Build the graph
        gpu_config = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.6)
        session_conf = tf.ConfigProto(gpu_options=gpu_config, intra_op_parallelism_threads=18, inter_op_parallelism_threads=18)
        tf.reset_default_graph()
        sess = tf.Session(config=session_conf)
                        
        # Create tf.variables for the three different datasets
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels), name='training_data')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='training_labels')
                            
        #tf.summary.image('input', tf_train_dataset, 6)
                            
        tf_valid_dataset = tf.constant(valid_dataset, name='validation_data')
        tf_valid_labels = tf.constant(valid_labels, name='validation_labels')
    
        # First layer is a convolution layer
        with tf.name_scope('conv2d_1'):
            layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name='W')
            layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth]), name='B')
    
            conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
            #tf.summary.histogram("weights", layer1_weights)
            #tf.summary.histogram("biases", layer1_biases)
            #tf.summary.histogram("activations", hidden)    
    
        # The reshape produces an input vector for the dense layer
        with tf.name_scope('reshape'):
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
    
        # Third layer is a dense layer
        with tf.name_scope('fc_1'):
            layer3_weights = tf.Variable(tf.truncated_normal([23*23*depth, num_hidden], stddev=0.1), name='W')
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B')
    
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    
            #tf.summary.histogram("weights", layer3_weights)
            #tf.summary.histogram("biases", layer3_biases)
            #tf.summary.histogram("activations", hidden)
    
        # Fourth layer is a dense output layer
        with tf.name_scope('fc_2'):
            layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1), name='W')
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='B')
    
            hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)
    
            #tf.summary.histogram("weights", layer4_weights)
            #tf.summary.histogram("biases", layer4_biases)
            #tf.summary.histogram("activations", output)
            
        # Fifth layer is a dense output layer
        with tf.name_scope('fc_3'):
            layer5_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), name='W')
            layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')
    
            output = tf.matmul(hidden, layer5_weights) + layer5_biases
    
            #tf.summary.histogram("weights", layer5_weights)
            #tf.summary.histogram("biases", layer5_biases)
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
            shape = pool_1.get_shape().as_list()
            reshape = tf.reshape(pool_1, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)
            valid_prediction = tf.nn.softmax(tf.matmul(hidden, layer5_weights) + layer5_biases)
                                
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
                #print('Auc: %.2f, %.2f' % (auc_val[0], auc_val[1]))
                #writer.add_summary(s, step)
                                    
                if step == 0:
                    stopping_auc = 0.0
                    sink_count = 0
                else:
                    if auc_val[0] > stopping_auc:
                        stopping_auc = auc_val[0]
                        sink_count = 0
                        if stopping_auc > best_auc:
                            saver.save(sess, os.path.join(folder_path, model_name))
                            best_auc = stopping_auc
                    else:
                        sink_count += 1
                print('St_auc: {}, sc: {},val: {}, Step: {}'.format(stopping_auc, sink_count, val*100, step))
                if sink_count == 5:
                    break   
    
        sess.close()
        dauer = time.time() - start
        early_stopped = True if step < num_steps-1 else False
        with open(os.path.join(folder_path, model_name+'_Hyperparameter.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([learning_rate, batch_size, patch_size, depth, num_hidden, val*100, stopping_auc, step, early_stopped, dauer, title_name])

    except:
        sess.close()
