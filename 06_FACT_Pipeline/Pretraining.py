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

mc_data_path = '/fhgfs/users/jbehnken/01_Data/01c_MC_Data_flat' # Path to preprocessed data
num_files = 500 # Number of files to load - 1 file = 1000 events
events_in_validation = 10000
number_of_nets = 20

save_model_path = '/fhgfs/users/jbehnken/01_Data/04_Models'
model_name = 'pre-cccfff'
title_name = 'with_5_100_pre_flat_c'

# Hyperparameter for the model (fit manually)
num_labels = 2 # gamma or proton
num_channels = 1 # it is a greyscale image

num_steps_pretraining = 5000
num_steps_final = 100000

learning_rate = 0.001
batch_size = np.random.randint(64, 257, size=number_of_nets)
patch_size = np.random.randint(0, 2, size=number_of_nets)*2 +3

depth_c1 = np.random.randint(4, 12, size=number_of_nets)
depth_c2 = np.random.randint(4, 12, size=number_of_nets)+depth_c1
depth_c3 = np.random.randint(4, 12, size=number_of_nets)+depth_c2

num_hidden_f1 = np.random.randint(10, 60, size=number_of_nets)
num_hidden_f2 = num_hidden_f1

dropout_rate_c = 0.9
dropout_rate_c_output = 0.75
dropout_rate_f = 0.5

trainable = True


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

test_dataset = pic[p][:events_in_validation]
test_labels = lab[p][:events_in_validation]
train_dataset = pic[p][events_in_validation:]
train_labels = lab[p][events_in_validation:]
del p, pic, lab
print('Data loaded')


def createFolderstructure():
    # Creating folder structure
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

    models_path = os.path.join(folder_path, 'models_folder')
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    run_folders = os.listdir(models_path)
    if len(run_folders)==0:
        count = [0]
    else:
        count = [int(folder.split('_')[0]) for folder in run_folders]

    run_path = os.path.join(models_path, str(max(count)+1)+'_'+title_name)
    os.mkdir(run_path)
    return run_path, folder_path, count


def training(steps):
    print('Layer {} training:'.format(iteration))
    for step in range(steps+1):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Creating a feed_dict to train the model on in this step
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

        opt = sess.run(optimizer, feed_dict=feed_dict)
        
        # Updating the output to stay in touch with the training process
        if (step % 1000 == 0):
            [acc, auc, s] = sess.run([test_accuracy, test_auc, summ], feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels})
            writer.add_summary(s, step)

            auc_now = auc[1]                        
            if step == 0:
                stopping_auc = 0.0
                sink_count = 0
            else:
                if auc_now > stopping_auc:
                    stopping_auc = auc_now
                    sink_count = 0
                    saver.save(sess, os.path.join(run_path, 'First_Layer'))
                else:
                    sink_count += 1
            print('St_auc: {}, sc: {},val: {}, Step: {}'.format(stopping_auc, sink_count, acc*100, step))
            if sink_count == 5:
                break
    return acc, stopping_auc, step


for batch_size, patch_size, depth_c1, depth_c2, depth_c3, num_hidden_f1, num_hidden_f2 in zip(batch_size, patch_size, depth_c1, depth_c2, depth_c3, num_hidden_f1, num_hidden_f2):
    hparams = '_bs={}_ps={}_d1={}_d2={}_d3={}_nh1={}_nh2={}'.format(batch_size, patch_size, depth_c1, depth_c2, depth_c3, num_hidden_f1, num_hidden_f2)
    run_path, folder_path, count = createFolderstructure()
        
    gpu_config = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
    session_conf = tf.ConfigProto(gpu_options=gpu_config, intra_op_parallelism_threads=18, inter_op_parallelism_threads=18)
    
    start = time.time()
    

    weights_1 = []
    biases_1 = []
    
    iteration = 1
    tf.reset_default_graph()
    with tf.Session(config=session_conf) as sess:
        print('Session {} created'.format(iteration))
        
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels), name='train_data')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='train_labels')
        tf.summary.image('input', tf_train_dataset, 6)

        tf_test_dataset = tf.constant(test_dataset, name='test_data')
        tf_test_labels = tf.constant(test_labels, name='test_labels')
        
        # First layer is a convolution layer
        with tf.name_scope('{}_conv2d_1'.format(iteration)):
            layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth_c1], stddev=0.1), name='W_1')
            layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth_c1]), name='B_1')

            conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c_output)

            tf.summary.histogram("weights", layer1_weights)
            tf.summary.histogram("biases", layer1_biases)
            tf.summary.histogram("activations", hidden)

        # The reshape produces an input vector for the dense layer
        with tf.name_scope('{}_reshape'.format(iteration)):
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

        # Output layer is a dense layer
        with tf.name_scope('{}_Output'.format(iteration)):
            output_weights = tf.Variable(tf.truncated_normal([23*23*depth_c1, num_labels], stddev=0.1), name='W')
            output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')

            output = tf.matmul(reshape, output_weights) + output_biases

            tf.summary.histogram("weights", output_weights)
            tf.summary.histogram("biases", output_biases)
            tf.summary.histogram("activations", output)

        # Computing the loss of the model
        with tf.name_scope('{}_loss'.format(iteration)):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')
            tf.summary.scalar("loss", loss)

        # Optimizing the model
        with tf.name_scope('{}_optimizer'.format(iteration)):
            optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(iteration)).minimize(loss)

        # Predictions for the training, validation, and test data
        with tf.name_scope('{}_prediction'.format(iteration)):
            train_prediction = tf.nn.softmax(output)
            
        # Evaluating the network: accuracy
        with tf.name_scope('{}_test'.format(iteration)):
            pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_test_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = pool_1.get_shape().as_list()
            reshape = tf.reshape(pool_1, [shape[0], shape[1] * shape[2] * shape[3]])
            test_prediction = tf.nn.softmax(tf.matmul(reshape, output_weights) + output_biases)

            correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(test_labels, 1))
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('test_accuracy', test_accuracy)                        

        # Evaluating the network: auc
        with tf.name_scope('{}_auc'.format(iteration)):
            test_auc = tf.metrics.auc(labels=tf_test_labels, predictions=test_prediction, curve='ROC')
            tf.summary.scalar('test_auc_0', test_auc[0])
            tf.summary.scalar('test_auc_1', test_auc[1])
        print('Layers created')
            
            
        summ = tf.summary.merge_all()
        saver = tf.train.Saver({"weight_1":layer1_weights, "bias_1":layer1_biases})

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        writer = tf.summary.FileWriter(os.path.join(run_path, 'First_Layer'+hparams))
        writer.add_graph(sess.graph)

        training(num_steps_pretraining)
        
        weights_1.append(layer1_weights.eval())
        biases_1.append(layer1_biases.eval())

        
    
    weights_2 = []
    biases_2 = []
    
    iteration = 2
    tf.reset_default_graph()
    with tf.Session(config=session_conf) as sess:
        print('Session {} created'.format(iteration))                    
            
        # Create tf.variables for the three different datasets
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels), name='train_data')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='train_labels')

        tf.summary.image('input', tf_train_dataset, 6)

        tf_test_dataset = tf.constant(test_dataset, name='test_data')
        tf_test_labels = tf.constant(test_labels, name='test_labels')
        
        # First layer is a convolution layer
        with tf.name_scope('{}_conv2d_1'.format(iteration)):
            init_w_1 = tf.constant(weights_1[0])
            layer1_weights = tf.get_variable('W_1', initializer=init_w_1, trainable=trainable)
            init_b_1 = tf.constant(biases_1[0])
            layer1_biases = tf.get_variable('B_1', initializer=init_b_1, trainable=trainable)
        
            conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c)

            tf.summary.histogram("weights", layer1_weights)
            tf.summary.histogram("biases", layer1_biases)
            tf.summary.histogram("activations", hidden)

        # Second layer is a convolution layer
        with tf.name_scope('{}_conv2d_2'.format(iteration)):
            layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_c1, depth_c2], stddev=0.1), name='W')
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth_c2]), name='B')

            conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME') 
            hidden = tf.nn.relu(conv + layer2_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c_output)

            tf.summary.histogram("weights", layer2_weights)
            tf.summary.histogram("biases", layer2_biases)
            tf.summary.histogram("activations", hidden)

        # The reshape produces an input vector for the dense layer
        with tf.name_scope('{}_reshape'.format(iteration)):
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

        # Output layer is a dense layer
        with tf.name_scope('{}_Output'.format(iteration)):
            output_weights = tf.Variable(tf.truncated_normal([12*12*depth_c2, num_labels], stddev=0.1), name='W')
            output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')

            output = tf.matmul(reshape, output_weights) + output_biases

            tf.summary.histogram("weights", output_weights)
            tf.summary.histogram("biases", output_biases)
            tf.summary.histogram("activations", output)

        # Computing the loss of the model
        with tf.name_scope('{}_loss'.format(iteration)):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')
            tf.summary.scalar("loss", loss)

        # Optimizing the model
        with tf.name_scope('{}_optimizer'.format(iteration)):
            optimizer = tf.train.AdamOptimizer(learning_rate, name='{}_adam'.format(iteration)).minimize(loss)

        # Predictions for the training, validation, and test data
        with tf.name_scope('{}_prediction'.format(iteration)):
            train_prediction = tf.nn.softmax(output)

        # Evaluating the network: accuracy
        with tf.name_scope('{}_test'.format(iteration)):
            pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_test_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = pool_2.get_shape().as_list()
            reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
            test_prediction = tf.nn.softmax(tf.matmul(reshape, output_weights) + output_biases)

            correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(test_labels, 1))
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('test_accuracy', test_accuracy)                        

        # Evaluating the network: auc
        with tf.name_scope('{}_auc'.format(iteration)):
            test_auc = tf.metrics.auc(labels=tf_test_labels, predictions=test_prediction, curve='ROC')
            tf.summary.scalar('test_auc_0', test_auc[0])
            tf.summary.scalar('test_auc_1', test_auc[1])
        print('Layers created')
        
            
        summ = tf.summary.merge_all()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        writer = tf.summary.FileWriter(os.path.join(run_path, 'Second_Layer'+hparams))
        writer.add_graph(sess.graph)

        training(num_steps_pretraining)
        
        weights_2.append(layer1_weights.eval())
        weights_2.append(layer2_weights.eval())
        biases_2.append(layer1_biases.eval())
        biases_2.append(layer2_biases.eval())
        
        
        
    
    weights_3 = []
    biases_3 = []
    
    iteration = 3
    tf.reset_default_graph()
    with tf.Session(config=session_conf) as sess:
        print('Session {} created'.format(iteration))
            
        # Create tf.variables for the three different datasets
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels), name='train_data')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='train_labels')

        tf.summary.image('input', tf_train_dataset, 6)

        tf_test_dataset = tf.constant(test_dataset, name='test_data')
        tf_test_labels = tf.constant(test_labels, name='test_labels')
        
        # First layer is a convolution layer
        with tf.name_scope('{}_conv2d_1'.format(iteration)):
            init_w_1 = tf.constant(weights_2[0])
            layer1_weights = tf.get_variable('W_1', initializer=init_w_1, trainable=trainable)
            init_b_1 = tf.constant(biases_2[0])
            layer1_biases = tf.get_variable('B_1', initializer=init_b_1, trainable=trainable)
            
            conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c)

            tf.summary.histogram("weights", layer1_weights)
            tf.summary.histogram("biases", layer1_biases)
            tf.summary.histogram("activations", hidden)

        # Second layer is a convolution layer
        with tf.name_scope('{}_conv2d_2'.format(iteration)):
            init_w_2 = tf.constant(weights_2[1])
            layer2_weights = tf.get_variable('W_2', initializer=init_w_2, trainable=trainable)
            init_b_2 = tf.constant(biases_2[1])
            layer2_biases = tf.get_variable('B_2', initializer=init_b_2, trainable=trainable)
            
            conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME') 
            hidden = tf.nn.relu(conv + layer2_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c)

            tf.summary.histogram("weights", layer2_weights)
            tf.summary.histogram("biases", layer2_biases)
            tf.summary.histogram("activations", hidden)

        # Third layer is a convolution layer
        with tf.name_scope('{}_conv2d_3'.format(iteration)):
            layer3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_c2, depth_c3], stddev=0.1), name='W')
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[depth_c3]), name='B')

            conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME') 
            hidden = tf.nn.relu(conv + layer3_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c_output)

            tf.summary.histogram("weights", layer3_weights)
            tf.summary.histogram("biases", layer3_biases)
            tf.summary.histogram("activations", hidden)

        # The reshape produces an input vector for the dense layer
        with tf.name_scope('{}_reshape'.format(iteration)):
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

        # Output layer is a dense layer
        with tf.name_scope('{}_Output'.format(iteration)):
            output_weights = tf.Variable(tf.truncated_normal([6*6*depth_c3, num_labels], stddev=0.1), name='W')
            output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')

            output = tf.matmul(reshape, output_weights) + output_biases

            tf.summary.histogram("weights", output_weights)
            tf.summary.histogram("biases", output_biases)
            tf.summary.histogram("activations", output)

        # Computing the loss of the model
        with tf.name_scope('{}_loss'.format(iteration)):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')
            tf.summary.scalar("loss", loss)

        # Optimizing the model
        with tf.name_scope('{}_optimizer'.format(iteration)):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data
        with tf.name_scope('{}_prediction'.format(iteration)):
            train_prediction = tf.nn.softmax(output)

        # Evaluating the network: accuracy
        with tf.name_scope('{}_test'.format(iteration)):
            pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_test_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = pool_3.get_shape().as_list()
            reshape = tf.reshape(pool_3, [shape[0], shape[1] * shape[2] * shape[3]])
            test_prediction = tf.nn.softmax(tf.matmul(reshape, output_weights) + output_biases)

            correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(test_labels, 1))
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('test_accuracy', test_accuracy)                        

        # Evaluating the network: auc
        with tf.name_scope('{}_auc'.format(iteration)):
            test_auc = tf.metrics.auc(labels=tf_test_labels, predictions=test_prediction, curve='ROC')
            tf.summary.scalar('test_auc_0', test_auc[0])
            tf.summary.scalar('test_auc_1', test_auc[1])
        print('Layers created')
            
            
        summ = tf.summary.merge_all()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        writer = tf.summary.FileWriter(os.path.join(run_path, 'Third_Layer'+hparams))
        writer.add_graph(sess.graph)

        training(num_steps_pretraining)
        
        weights_3.append(layer1_weights.eval())
        weights_3.append(layer2_weights.eval())
        weights_3.append(layer3_weights.eval())
        biases_3.append(layer1_biases.eval())
        biases_3.append(layer2_biases.eval())
        biases_3.append(layer3_biases.eval())
        
        
        
    weights_4 = []
    biases_4 = []
    
    iteration = 4
    tf.reset_default_graph()
    with tf.Session(config=session_conf) as sess:
        print('Session created')
            
        # Create tf.variables for the three different datasets
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels), name='train_data')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='train_labels')

        tf.summary.image('input', tf_train_dataset, 6)

        tf_test_dataset = tf.constant(test_dataset, name='test_data')
        tf_test_labels = tf.constant(test_labels, name='test_labels')
        
        # First layer is a convolution layer
        with tf.name_scope('{}_conv2d_1'.format(iteration)):
            init_w_1 = tf.constant(weights_3[0])
            layer1_weights = tf.get_variable('W_1', initializer=init_w_1, trainable=trainable)
            init_b_1 = tf.constant(biases_3[0])
            layer1_biases = tf.get_variable('B_1', initializer=init_b_1, trainable=trainable)
            
            conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c)

            tf.summary.histogram("weights", layer1_weights)
            tf.summary.histogram("biases", layer1_biases)
            tf.summary.histogram("activations", hidden)

        # Second layer is a convolution layer
        with tf.name_scope('{}_conv2d_2'.format(iteration)):
            init_w_2 = tf.constant(weights_3[1])
            layer2_weights = tf.get_variable('W_2', initializer=init_w_2, trainable=trainable)
            init_b_2 = tf.constant(biases_3[1])
            layer2_biases = tf.get_variable('B_2', initializer=init_b_2, trainable=trainable)
            
            conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME') 
            hidden = tf.nn.relu(conv + layer2_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c)

            tf.summary.histogram("weights", layer2_weights)
            tf.summary.histogram("biases", layer2_biases)
            tf.summary.histogram("activations", hidden)

        # Third layer is a convolution layer
        with tf.name_scope('{}_conv2d_3'.format(iteration)):
            init_w_3 = tf.constant(weights_3[2])
            layer3_weights = tf.get_variable('W_3', initializer=init_w_3, trainable=trainable)
            init_b_3 = tf.constant(biases_3[2])
            layer3_biases = tf.get_variable('B_3', initializer=init_b_3, trainable=trainable)
            
            conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME') 
            hidden = tf.nn.relu(conv + layer3_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c_output)

            tf.summary.histogram("weights", layer3_weights)
            tf.summary.histogram("biases", layer3_biases)
            tf.summary.histogram("activations", hidden)

        # The reshape produces an input vector for the dense layer
        with tf.name_scope('{}_reshape'.format(iteration)):
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

        # Fourth layer is a dense layer
        with tf.name_scope('{}_fc_1'.format(iteration)):
            layer4_weights = tf.Variable(tf.truncated_normal([6*6*depth_c3, num_hidden_f1], stddev=0.1), name='W')
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_f1]), name='B')

            hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
            hidden = tf.nn.dropout(hidden, dropout_rate_f)

            tf.summary.histogram("weights", layer4_weights)
            tf.summary.histogram("biases", layer4_biases)
            tf.summary.histogram("activations", hidden)

        # Output layer is a dense layer
        with tf.name_scope('{}_Output'.format(iteration)):
            output_weights = tf.Variable(tf.truncated_normal([num_hidden_f1, num_labels], stddev=0.1), name='W')
            output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')

            output = tf.matmul(hidden, output_weights) + output_biases

            tf.summary.histogram("weights", output_weights)
            tf.summary.histogram("biases", output_biases)
            tf.summary.histogram("activations", output)

        # Computing the loss of the model
        with tf.name_scope('{}_loss'.format(iteration)):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')
            tf.summary.scalar("loss", loss)

        # Optimizing the model
        with tf.name_scope('{}_optimizer'.format(iteration)):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data
        with tf.name_scope('{}_prediction'.format(iteration)):
            train_prediction = tf.nn.softmax(output)

        # Evaluating the network: accuracy
        with tf.name_scope('{}_test'.format(iteration)):
            pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_test_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = pool_3.get_shape().as_list()
            reshape = tf.reshape(pool_3, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden_1 = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
            test_prediction = tf.nn.softmax(tf.matmul(hidden_1, output_weights) + output_biases)

            correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(test_labels, 1))
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('test_accuracy', test_accuracy)                        

        # Evaluating the network: auc
        with tf.name_scope('{}_auc'.format(iteration)):
            test_auc = tf.metrics.auc(labels=tf_test_labels, predictions=test_prediction, curve='ROC')
            tf.summary.scalar('test_auc_0', test_auc[0])
            tf.summary.scalar('test_auc_1', test_auc[1])
        print('Layers created')
            
            
        summ = tf.summary.merge_all()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        writer = tf.summary.FileWriter(os.path.join(run_path, 'Fourth_Layer'+hparams))
        writer.add_graph(sess.graph)
        
        training(num_steps_pretraining)
        
        weights_4.append(layer1_weights.eval())
        weights_4.append(layer2_weights.eval())
        weights_4.append(layer3_weights.eval())
        weights_4.append(layer4_weights.eval())
        biases_4.append(layer1_biases.eval())
        biases_4.append(layer2_biases.eval())
        biases_4.append(layer3_biases.eval())
        biases_4.append(layer4_biases.eval())
        
        
        
    iteration = 5
    tf.reset_default_graph()    
    with tf.Session(config=session_conf) as sess:
        print('Session created')
            
        # Create tf.variables for the three different datasets
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels), name='train_data')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='train_labels')

        tf.summary.image('input', tf_train_dataset, 6)

        tf_test_dataset = tf.constant(test_dataset, name='test_data')
        tf_test_labels = tf.constant(test_labels, name='test_labels')
        
        # First layer is a convolution layer
        with tf.name_scope('{}_conv2d_1'.format(iteration)):
            init_w_1 = tf.constant(weights_4[0])
            layer1_weights = tf.get_variable('W_1', initializer=init_w_1, trainable=trainable)
            init_b_1 = tf.constant(biases_4[0])
            layer1_biases = tf.get_variable('B_1', initializer=init_b_1, trainable=trainable)
            
            conv = tf.nn.conv2d(tf_train_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c)

            tf.summary.histogram("weights", layer1_weights)
            tf.summary.histogram("biases", layer1_biases)
            tf.summary.histogram("activations", hidden)

        # Second layer is a convolution layer
        with tf.name_scope('{}_conv2d_2'.format(iteration)):
            init_w_2 = tf.constant(weights_4[1])
            layer2_weights = tf.get_variable('W_2', initializer=init_w_2, trainable=trainable)
            init_b_2 = tf.constant(biases_4[1])
            layer2_biases = tf.get_variable('B_2', initializer=init_b_2, trainable=trainable)
            
            conv = tf.nn.conv2d(pool, layer2_weights, [1, 1, 1, 1], padding='SAME') 
            hidden = tf.nn.relu(conv + layer2_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c)

            tf.summary.histogram("weights", layer2_weights)
            tf.summary.histogram("biases", layer2_biases)
            tf.summary.histogram("activations", hidden)

        # Third layer is a convolution layer
        with tf.name_scope('{}_conv2d_3'.format(iteration)):
            init_w_3 = tf.constant(weights_4[2])
            layer3_weights = tf.get_variable('W_3', initializer=init_w_3, trainable=trainable)
            init_b_3 = tf.constant(biases_4[2])
            layer3_biases = tf.get_variable('B_3', initializer=init_b_3, trainable=trainable)
            
            conv = tf.nn.conv2d(pool, layer3_weights, [1, 1, 1, 1], padding='SAME') 
            hidden = tf.nn.relu(conv + layer3_biases)
            pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool = tf.nn.dropout(pool, dropout_rate_c_output)

            tf.summary.histogram("weights", layer3_weights)
            tf.summary.histogram("biases", layer3_biases)
            tf.summary.histogram("activations", hidden)

        # The reshape produces an input vector for the dense layer
        with tf.name_scope('{}_reshape'.format(iteration)):
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])

        # Fourth layer is a dense layer
        with tf.name_scope('{}_fc_1'.format(iteration)):
            init_w_4 = tf.constant(weights_4[3])
            layer4_weights = tf.get_variable('W_4', initializer=init_w_4, trainable=trainable)
            init_b_4 = tf.constant(biases_4[3])
            layer4_biases = tf.get_variable('B_4', initializer=init_b_4, trainable=trainable)
            
            hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
            hidden = tf.nn.dropout(hidden, dropout_rate_f)

            tf.summary.histogram("weights", layer4_weights)
            tf.summary.histogram("biases", layer4_biases)
            tf.summary.histogram("activations", hidden)

        # Fifth layer is a dense layer
        with tf.name_scope('{}_fc_2'.format(iteration)):
            layer5_weights = tf.Variable(tf.truncated_normal([num_hidden_f1, num_hidden_f2], stddev=0.1), name='W')
            layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_f2]), name='B')

            hidden = tf.nn.relu(tf.matmul(hidden, layer5_weights) + layer5_biases)
            hidden = tf.nn.dropout(hidden, dropout_rate_f)

            tf.summary.histogram("weights", layer5_weights)
            tf.summary.histogram("biases", layer5_biases)
            tf.summary.histogram("activations", hidden)

        # Output layer is a dense layer
        with tf.name_scope('{}_Output'.format(iteration)):
            output_weights = tf.Variable(tf.truncated_normal([num_hidden_f2, num_labels], stddev=0.1), name='W')
            output_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B')

            output = tf.matmul(hidden, output_weights) + output_biases

            tf.summary.histogram("weights", output_weights)
            tf.summary.histogram("biases", output_biases)
            tf.summary.histogram("activations", output)

        # Computing the loss of the model
        with tf.name_scope('{}_loss'.format(iteration)):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_train_labels), name='loss')
            tf.summary.scalar("loss", loss)

        # Optimizing the model
        with tf.name_scope('{}_optimizer'.format(iteration)):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data
        with tf.name_scope('{}_prediction'.format(iteration)):
            train_prediction = tf.nn.softmax(output)

        # Evaluating the network: accuracy
        with tf.name_scope('{}_test'.format(iteration)):
            pool_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(tf_test_dataset, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool_2, layer3_weights, [1, 1, 1, 1], padding='SAME') + layer3_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = pool_3.get_shape().as_list()
            reshape = tf.reshape(pool_3, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden_1 = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
            hidden_2 = tf.nn.relu(tf.matmul(hidden_1, layer5_weights) + layer5_biases)
            test_prediction = tf.nn.softmax(tf.matmul(hidden_2, output_weights) + output_biases)

            correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(test_labels, 1))
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('test_accuracy', test_accuracy)                        

        # Evaluating the network: auc
        with tf.name_scope('{}_auc'.format(iteration)):
            test_auc = tf.metrics.auc(labels=tf_test_labels, predictions=test_prediction, curve='ROC')
            tf.summary.scalar('test_auc_0', test_auc[0])
            tf.summary.scalar('test_auc_1', test_auc[1])
        print('Layers created')
            
            
        summ = tf.summary.merge_all()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        writer = tf.summary.FileWriter(os.path.join(run_path, 'Fifth_Layer'+hparams))
        writer.add_graph(sess.graph)
        acc, stopping_auc, step = training(num_steps_final)

        dauer = time.time() - start
        early_stopped = True if step < num_steps_final-1 else False
        
        with open(os.path.join(folder_path, model_name+'_Hyperparameter.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([learning_rate, batch_size, patch_size, [depth_c1, depth_c2, depth_c3], [num_hidden_f1, num_hidden_f2], acc*100, stopping_auc, step, early_stopped, dauer, str(max(count)+1)+'_'+title_name])
print('Finished!')
