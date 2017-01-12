
# coding: utf-8

# In[3]:

import os, random
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


# In[4]:

tinyImageNetDir = "/home/devyhia/vgg"
X, Y = np.load("{}/X.npy".format(tinyImageNetDir)), np.load("{}/y.npy".format(tinyImageNetDir))
Xt, Yt = np.load("{}/Xt.npy".format(tinyImageNetDir)), np.load("{}/yt.npy".format(tinyImageNetDir))


# In[11]:

tf.reset_default_graph()

# Parameters
learning_rate = 0.001
batch_size = 50
display_step = 25
epochs = 100
depth = 5

# Network Parameters
n_input = 64 * 3 # MNIST data input (img shape: 28*28)
n_steps = 64 # timesteps
n_hidden = 512 # hidden layer num of features
n_classes = 100 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float32", [None, n_steps, n_input])
y = tf.placeholder("float32", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[12]:

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    multi_cells = rnn_cell.MultiRNNCell([lstm_cell] * depth, state_is_tuple=True)

    # Get lstm cell output
    outputs, states = rnn.rnn(multi_cells, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# In[13]:

def __iterate_minibatches(_X,_y, size):
    if _X.shape[0] % size > 0:
        raise "The minibatch size should be a divisor of the batch size."

    idx = np.arange(_X.shape[0]).astype(np.int32)
    np.random.shuffle(idx) # in-place shuffling
    for i in range(_X.shape[0] / size):
        # To randomize the minibatches every time
        _idx = idx[i*size:(i+1)*size]
        _X_small = _X[_idx]
        _y_small = _y[_idx]
        yield _X_small, _y_small


# In[14]:

def calculate_loss(sess, Xt, yt, size=1000, step=10):
        fc3ls = None
        sample_idx = random.sample(range(0, Xt.shape[0]), size)
        for i in range(size / step):
            [fc3l] = sess.run([pred], feed_dict={x: Xt[sample_idx[i*step:(i+1)*step]], y: yt[sample_idx[i*step:(i+1)*step]]})
            if i == 0:
                fc3ls = fc3l
            else:
                fc3ls = np.vstack((fc3ls, fc3l))

        loss, acc = sess.run([cost, accuracy], feed_dict={pred: fc3ls, y: yt[sample_idx]})

        return loss, acc


# In[10]:

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    rnn_shape = (-1, n_steps, n_input)
    for ep in range(epochs):
        print("==== EPOCH {} ====".format(ep))
        step = 1
        for _X, _Y in __iterate_minibatches(X, Y, batch_size):
            _X = _X.reshape(rnn_shape)
            sess.run(optimizer, feed_dict={x: _X, y: _Y})
            if step % display_step == 0:
                loss, acc = calculate_loss(sess, Xt.reshape(rnn_shape), Yt)
                print("Iter " + str(step) + ", Minibatch Loss= " +                       "{:.4f}".format(loss) + ", Training Accuracy= " +                       "{:.4f}".format(acc))
            step += 1
        
        loss, acc = calculate_loss(sess, Xt.reshape(rnn_shape), Yt, size=Xt.shape[0])
        print("====================================")
        print("Epoch {}: Loss={} Acc={}".format(ep, loss, acc))
        print("====================================")


# In[ ]:



