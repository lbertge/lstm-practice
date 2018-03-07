import tensorflow as tf
import numpy as np



"""
    ⭐ Train an LSTM to solve the XOR problem: that is, given a sequence of
    bits, determine its parity. The LSTM should consume the sequence, one bit at a
    time, and then output the correct answer at the sequence’s end. Test the two
    approaches below:

    Generate a dataset of random 100,000 binary strings of length 50. Train the
    LSTM; what performance do you get?  Generate a dataset of random 100,000
    binary strings, where the length of each string is independently and
    randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What
    explains the difference?
"""

TOTAL_SIZE = 100000
DATA_SIZE = int(0.6 * TOTAL_SIZE)
VAL_SIZE = int(0.2 * TOTAL_SIZE)
TEST_SIZE = int(0.2 * TOTAL_SIZE)

BATCH_SIZE = 1000

LENGTH = 50
NUM_INPUT = 1
NUM_UNITS = 128
NUM_CLASSES = 2
RATE = 0.01
EPOCH = 4000
PRINT_EVERY_EPOCH = EPOCH / 10

STOPPING_CRIT = 0.05

# Creating random sequences and labels

def create_xor_data(size):
    X_ = np.floor(np.random.rand(size, LENGTH) * 2)
    X_ = X_.reshape(size, LENGTH, 1)
    Y_ = np.array(np.mod(np.sum(X_, 1), 2.0))
    Y_ = np.squeeze(Y_)
    Y_logits = np.zeros((len(Y_), NUM_CLASSES))
    Y_logits[np.arange(Y_.size), Y_.astype('int64')] = 1
    return X_, Y_, Y_logits

def next_batch_shuffle(a, b, batch_size = None):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if batch_size:
        p = p[:batch_size]
    return a[p], b[p]

train_X, train_Y, train_logits = create_xor_data(DATA_SIZE)
valid_X, valid_Y, valid_logits = create_xor_data(VAL_SIZE)
test_X, test_Y, test_logits = create_xor_data(TEST_SIZE)

# build LSTM
weights = {
    'out': tf.Variable(tf.random_normal([NUM_UNITS, NUM_CLASSES]))
}
biases = {
    'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
}

X = tf.placeholder("float", [None, LENGTH, NUM_INPUT])
Y = tf.placeholder("float", [None, NUM_CLASSES])

def network(x, weights, biases):
    x = tf.unstack(x, LENGTH, axis=1)
    lstm = tf.contrib.rnn.BasicLSTMCell(NUM_UNITS)
    output, states = tf.contrib.rnn.static_rnn(lstm, x, dtype=tf.float32)

    return tf.matmul(output[-1], weights['out']) + biases['out']

logits = network(X, weights, biases)
pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
opt = tf.train.AdamOptimizer(learning_rate = RATE)
train = opt.minimize(loss)

correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# training
with tf.Session() as sess:
    sess.run(init)

    for step in range(EPOCH):
        train_batch = next_batch_shuffle(train_X, train_logits, batch_size = BATCH_SIZE)
        train_loss = sess.run(train, feed_dict={X: train_batch[0], Y: train_batch[1]})

        a = None
        if step % PRINT_EVERY_EPOCH == 0:
            l, a = sess.run([loss, acc], feed_dict={X: valid_X, Y: valid_logits})
            print("Step " + str(step) + ", Valid Loss= " + \
                    "{:.3f}".format(l) + ", Valid Acc= " + \
                    "{:.3f}".format(a))
        if a and 1 - a < STOPPING_CRIT:
            print("Epoch " + str(step) + "; stopping here, I have probably memorized the set")
            break

    print("Test Acc:", sess.run(acc, feed_dict={X: test_X, Y: test_logits}))
    print("Train Acc:", sess.run(acc, feed_dict={X: train_X, Y: train_logits}))
    print("Valid Acc:", sess.run(acc, feed_dict={X: valid_X, Y: valid_logits}))

    example_X = test_X[0]
    example_X = example_X.reshape((1, LENGTH, NUM_INPUT))
    example_logit = test_logits[0]
    example_logit = example_logit.reshape((1, NUM_CLASSES))
