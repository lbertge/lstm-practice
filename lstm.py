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

TOTAL_SIZE = 10
DATA_SIZE = int(0.6 * TOTAL_SIZE)
VAL_SIZE = int(0.2 * TOTAL_SIZE)
TEST_SIZE = int(0.2 * TOTAL_SIZE)

BATCH_SIZE = 1

LENGTH = 2
NUM_INPUT = 1
NUM_UNITS = 128
NUM_CLASSES = 2
RATE = 0.001
EPOCH = 2000
PRINT_EVERY_EPOCH = EPOCH / 10

STOPPING_CRIT = 0.05

# Creating random sequences and labels
one_hot = lambda x: [0., 1.] if x == 1 else [1., 0.]
f = np.vectorize(one_hot)

train_X = np.floor(np.random.rand(DATA_SIZE, LENGTH) * 2)
train_X = train_X.reshape(DATA_SIZE, LENGTH, 1)
train_Y = np.array(np.mod(np.sum(train_X, 1), 2.0))
train_Y = np.squeeze(train_Y)
train_logits = np.zeros((len(train_Y), NUM_CLASSES))
train_logits[np.arange(train_Y.size), train_Y.astype('int64')] = 1


valid_X = np.floor(np.random.rand(VAL_SIZE, LENGTH) * 2)
valid_X = valid_X.reshape(VAL_SIZE, LENGTH, 1)
valid_Y = np.array(np.mod(np.sum(valid_X, 1), 2.0))
valid_Y = np.squeeze(valid_Y)
valid_logits = np.zeros((len(valid_Y), NUM_CLASSES))
valid_logits[np.arange(valid_Y.size), valid_Y.astype('int64')] = 1

test_X = np.floor(np.random.rand(TEST_SIZE, LENGTH) * 2)
test_X = test_X.reshape(TEST_SIZE, LENGTH, 1)
test_Y = np.array(np.mod(np.sum(test_X, 1), 2.0))
test_Y = np.squeeze(test_Y)
test_logits = np.zeros((len(test_Y), NUM_CLASSES))
test_logits[np.arange(test_Y.size), test_Y.astype('int64')] = 1

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
        train_loss = sess.run(train, feed_dict={X: train_X, Y: train_logits})

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

print(valid_X.T)
print(train_X.T)
print(test_X.T)
