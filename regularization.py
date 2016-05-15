#!/usr/bin/python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# define SGD
batch_size = 128
# batch_size = 30
hidden_nodes = 1024
beta = 0.001
decay_step = train_dataset.shape[0]/5
print('decay_step', decay_step)

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, hidden_nodes]))
    biases1 = tf.Variable(tf.zeros([hidden_nodes]))
    weights2 = tf.Variable(
        tf.truncated_normal([hidden_nodes, num_labels]))
    biases2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits1 = tf.matmul(tf_train_dataset, weights1) + biases1
    logits_relu1 = tf.nn.relu(logits1)
    # logits2 = tf.nn.dropout(logits2, 0.5)
    logits_final = tf.matmul(logits_relu1, weights2)+biases2

    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_final, tf_train_labels))
    loss2 = beta*(tf.nn.l2_loss(weights2)+tf.nn.l2_loss(weights1))

    loss = loss1

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # batch_step = tf.Variable(0)  # count the number of steps taken.
    # learning_rate = tf.train.exponential_decay(0.01, batch_step * batch_size, decay_step, 0.9)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits_final)

    # Prediction/Test computation.
   # logits_valid1 = tf.matmul(tf_valid_dataset, weights1) + biases1
   # logits_valid1 = tf.nn.relu(logits_valid1)
   # # logits_valid2 = tf.nn.dropout(logits_valid2, 0.5)
   # logits_valid2 = tf.matmul(logits_valid1, weights2)+biases2
   # logits_valid2 = tf.nn.relu(logits_valid2)
   # logits_valid_final = tf.matmul(logits_valid2, weights3)+biases3
   # valid_prediction = tf.nn.softmax(logits_valid_final)
    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,weights1)+biases1),weights2)+biases2)
    # tf.matmul(tf_valid_dataset, weights) + biases)

   # logits_test1 = tf.matmul(tf_test_dataset, weights1) + biases1
   # logits_test1 = tf.nn.relu(logits_test1)
   # # logits_test2 = tf.nn.dropout(logits_test2, 0.5)
   # logits_test2 = tf.matmul(logits_test1, weights2)+biases2
   # logits_test2 = tf.nn.relu(logits_test2)
   # logits_test_final = tf.matmul(logits_test2, weights3)+biases3
   # test_prediction = tf.nn.softmax(logits_test_final)
    test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1)+biases1), weights2)+biases2)
    # tf.matmul(tf_test_dataset, weights) + biases)

# run SGD
num_steps = 3001
# batch_num = 5

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # offset = (step * batch_size) % ( batch_size * batch_num )
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions, l1, l2 = session.run(
            [optimizer, loss, train_prediction, loss1, loss2], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
            print("loss1: %f" % l1)
            print("loss2: %f" % l2)
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
