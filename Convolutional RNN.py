
# coding: utf-8

# In[15]:

import argparse

parser = argparse.ArgumentParser(description='RNN-CNN Network.')
parser.add_argument('--depth', default=1, help='Depth of the RNN network')
parser.add_argument('--hidden', default=256, help='Hidden units of the RNN network')
parser.add_argument('--gpu', default=3, help='GPU to use for train')
parser.add_argument('--name', default="rnn_model", help='Name of the RNN model to use for train')
args, unknown_args = parser.parse_known_args()


# In[1]:

import os

os.environ["gpu"] = str(args.gpu)

from Inception_V4 import *

config = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 1})

sess = tf.Session(config=config)
# saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())


# ## RNN Model

# In[2]:

def process_features(X):
    return X.transpose([0, 3, 1, 2]).reshape((-1, 1024, 17*17))


# In[3]:

from tensorflow.python.ops import rnn, rnn_cell


# In[4]:

print("Set graph ...")
tf.reset_default_graph()

# Parameters
learning_rate = 0.001
batch_size = 50
display_step = 25
epochs = 100
depth = int(args.depth)

# Network Parameters
n_input = 17 * 17 # MNIST data input (img shape: 28*28)
n_steps = 1024 # timesteps
n_hidden = int(args.hidden) # hidden layer num of features
n_classes = 100 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float32", [None, n_steps, n_input], name="x")
y = tf.placeholder("float32", [None, n_classes], name="y")

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[5]:

print("RNN() ...")
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


# In[6]:

print("Build lstm cells ...")
pred = RNN(x, weights, biases)


# In[7]:

print("Setup cost ...")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))


# In[8]:

print("Setup optimizer ...")
# Define loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[9]:

print("Setup evaluation metrics ...")
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[10]:

print("Setup initializer ...")
# Initializing the variables
init = tf.initialize_all_variables()


# In[11]:

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


# In[12]:

def calculate_loss(sess, model, size=20, step=1):
        fc3ls = []
        sample_idx = random.sample(range(0, 100), size)
        for i in sample_idx:
            Xt = np.load("features/{}.Mixed_6h.Xt.{}.npy".format(model, i))
            Xt = process_features(Xt)

            [fc3l] = sess.run([pred], feed_dict={x: Xt})
            fc3ls.append(fc3l)
        
        fc3ls = np.vstack(fc3ls)
        yt = np.vstack([np.load("features/{}.Mixed_6h.yt.{}.npy".format(model, i)) for i in sample_idx])
        loss, acc = sess.run([cost, accuracy], feed_dict={pred: fc3ls, y: yt})

        return loss, acc


# In[14]:

def update_screen(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


# In[25]:

def save_checkpoint(sess, saver, model, prev_acc, curr_acc):
    if curr_acc > prev_acc:
        saver.save(sess, "{}.tfmodel".format(model))
        print("+++ Saved model")


# In[19]:

rnn_model = args.name
cnn_model = "model1"


# In[18]:

print("Start session ...")
# Launch the graph
sess = tf.Session()
print("Init session ...")
sess.run(init)
rnn_shape = (-1, n_steps, n_input)


# In[21]:

print("Create Saver ...")
saver = tf.train.Saver()


# In[30]:

prev_acc = 0
for ep in range(epochs):
    print("==== EPOCH {} ====".format(ep))
    step = 1
    size = 1000
    random_index = random.sample(range(0, size), size)
    for i in random_index:
        update_screen("\rIter {}".format(step))
        _X = np.load("features/{}.Mixed_6h.X.{}.npy".format(cnn_model, i))
        _X = process_features(_X)

        _Y = np.load("features/{}.Mixed_6h.y.{}.npy".format(cnn_model, i))

        sess.run(optimizer, feed_dict={x: _X, y: _Y})
        if step % display_step == 0:
            loss, acc = calculate_loss(sess, cnn_model)
            print("\rIter " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss) + ", Training Accuracy= " +                   "{:.4f}".format(acc))
        step += 1

    loss, acc = calculate_loss(sess, cnn_model, size=100)
    print("====================================")
    print("Epoch {}: Loss={} Acc={}".format(ep, loss, acc))
    print("====================================")
    save_checkpoint(sess, saver, rnn_model, prev_acc, acc)
    prev_acc = acc


# In[ ]:



