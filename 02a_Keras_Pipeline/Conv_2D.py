# Import of every needed library
import pickle
import gzip
import numpy as np
import tensorflow as tf
import os
import random
from multiprocessing import Pool


# Path to preprocessed data
path = '/fhgfs/users/jbehnken/rand_Conv_Data'


# Load pickled data and split it into pictures and labels
def load_data(file):
	with gzip.open(path+'/'+file, 'rb') as f:
		data_dict = pickle.load(f)
	pic = data_dict['Image']
	lab = data_dict['Label']
	return (pic, lab)


# Pool-load pickled data and split it into pictures and labels (list)
p = Pool()
data = p.map(load_data, os.listdir(path))
pics, labs = zip(*data)
del data, p


# Concatenate the data to a single np.array
pic = np.concatenate(pics)
lab = np.concatenate(labs)
del pics, labs


# Randomize and split the data into train/validation/test dataset
p = np.random.permutation(len(pic))
valid_dataset = pic[p][:10000]
valid_labels = lab[p][:10000]
test_dataset = pic[p][10000:60000]
test_labels = lab[p][10000:60000]
train_dataset = pic[p][60000:]
train_labels = lab[p][60000:]
del p, pic, lab


# Compute the accuracy
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


# Hyperparameter for the model (fit manually)
num_labels = 2 # gamma or proton
num_channels = 1 # it is a greyscale image

num_steps = 200001
batch_size = 64
patch_size = 1
depth = 16
num_hidden = 64


# Create the Tensorflow-Graph
graph = tf.Graph()
with graph.as_default():
    
    # Create tf.variables for the three different datasets
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 46, 45, num_channels)) # [46,45] is the shape of the data matrix
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # Create tf.variables for weights and biases for every layer
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1)) # Shouldn't the depth increase?
    layer1_biases = tf.Variable(tf.zeros([depth]))
    
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth])) # Why are the biases different (zeros/constant)?
    
    layer3_weights = tf.Variable(tf.truncated_normal([2304, num_hidden], stddev=0.1)) # Where does 2304 come from?
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1)) # This seems to be the output layer
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
    # Create the relation between all layer-variables
    def model(data):
        # First layer is a convolution layer
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME') # 'SAME' means to fill imageborders with zeros
        hidden = tf.nn.relu(conv + layer1_biases)
        
        # Second layer is a convolution layer
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME') # [1,2,2,1] 1s are fixes, 2s are strides
        hidden = tf.nn.relu(conv + layer2_biases)
        
        # What does thie reshape do? Seems to compute the convolutioned layers shape
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        
        # Third layer is a dense layer
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        
        # Fourth layer is a dense layer
        return tf.matmul(hidden, layer4_weights) + layer4_biases # Flows right into the cross_entropy
  
    # Training and computing the loss of the model
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # Optimizing the model
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
    # Predictions for the training, validation, and test data
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

# This gathers the learning process
valid_acc = []

# Creating the session and computing the model
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
    
    # Iterating over num_setps 
	for step in range(num_steps):
        # Computing the offset to move over the training dataset
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size) # What does the modulo do?
        # Getting the batchdata
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        # Getting the batchlabels
		batch_labels = train_labels[offset:(offset + batch_size), :]
        # Creating a feed_dict to train the model on in this step
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        # Train the model for this step
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        # Updating the output to stay in touch with the training process
		if (step % 100 == 0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
			valid_acc.append(accuracy(valid_prediction.eval(), valid_labels))
    # Compute the final accuracy for the model
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

# Save the gathered validation accuracys
with open('Valid_acc_0.p', 'wb') as f:
	pickle.dump(valid_acc, f)
