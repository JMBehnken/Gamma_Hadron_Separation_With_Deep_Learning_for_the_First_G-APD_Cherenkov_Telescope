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
data = p.map(load_data, os.listdir(path)[:150])
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


# Hyperparameter for the model (fit manually)
num_labels = 2 # gamma or proton
num_channels = 1 # it is a greyscale image

num_steps = 10001
batch_size = 64
patch_size = 3
depth = 32
num_hidden = 512
learning_rate = 0.05
LOGDIR = '/home/jbehnken/mnist_tutorial/3'


def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([patch_size, patch_size, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def mnist_model(learning_rate, hparam):
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=(None, 46, 45, num_channels), name="x")
    tf.summary.image('input', x, 6)
    y = tf.placeholder(tf.float32, shape=(None, num_labels), name="labels") # None, ehemals batch_size

    conv1 = conv_layer(x, num_channels, depth, "conv1")
    conv_out = conv_layer(conv1, depth, 2*depth, "conv2")

    flattened = tf.reshape(conv_out, [-1, 12 * 12 * 2*depth])

    fc1 = fc_layer(flattened, 12 * 12 * 2*depth, num_hidden, "fc1")
    logits = fc_layer(fc1, num_hidden, num_labels, "fc2")


    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="loss")
        tf.summary.scalar("loss", loss)

    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    for i in range(num_steps):
        data_batch = train_dataset[i*batch_size:(i+1)*batch_size]
        labels_batch = train_labels[i*batch_size:(i+1)*batch_size]
        if i % 50 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: data_batch, y: labels_batch})
            writer.add_summary(s, i)
            print(i)
        sess.run(train_step, feed_dict={x: data_batch, y: labels_batch})
        
        
def make_hparam_string(learning_rate):
    conv_param = "conv=2"
    fc_param = "fc=2"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)



def main():
    # You can try adding some more learning rates
    for learning_rate in [1E-4]:

        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
        hparam = make_hparam_string(learning_rate)
        print('Starting run for %s' % hparam)

        # Actually run with the new settings
        mnist_model(learning_rate, hparam)
        
main()