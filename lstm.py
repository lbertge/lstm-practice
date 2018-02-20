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

TOTAL_SIZE = 100
TRAINING_SPLIT = 0.64
VALIDATION_SPLIT = 0.16
TEST_SPLIT = 0.2
BATCH_SIZE = 1

LENGTH = 50
NUM_INPUT = 1
NUM_UNITS = 512
NUM_CLASSES = 2
RATE = 0.001
EPOCH = 10


# Creating random sequences and labels
train = tf.floor(tf.random_uniform([int(TRAINING_SPLIT * TOTAL_SIZE), LENGTH]) * 2)
train_label = tf.mod(tf.reduce_sum(train, 1), tf.constant(2.0))
train_label = tf.map_fn(lambda x : [1., 0.] if x > 0 else [0., 1.], train_label)

valid = tf.floor(tf.random_uniform([int(VALIDATION_SPLIT * TOTAL_SIZE), LENGTH]) * 2)
valid_label = tf.mod(tf.reduce_sum(valid, 1), tf.constant(2.0))
valid_label = tf.map_fn(lambda x : [1., 0.] if x > 0 else [0., 1.], valid_label)

test = tf.floor(tf.random_uniform([int(TEST_SPLIT * TOTAL_SIZE), LENGTH]) * 2)
test_label = tf.mod(tf.reduce_sum(valid, 1), tf.constant(2.0))
test_label = tf.map_fn(lambda x : [1., 0.] if x > 0 else [0., 1.], test_label)

train_dataset = tf.data.Dataset.from_tensor_slices((train, train_label))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid, valid_label))
test_dataset = tf.data.Dataset.from_tensor_slices((test, test_label))

#Create an iterator for accessing elements in batches
train_batch = train_dataset.batch(BATCH_SIZE)
train_batch = train_batch.repeat(EPOCH)

valid_batch = valid_dataset.batch(BATCH_SIZE)
valid_batch = vaild_batch.repeat(EPOCH)

train_it = train_dataset.make_one_shot_iterator()
train_next = train_it.get_next()

valid_it = valid_dataset.make_one_shot_iterator()
valid_next = valid_it.get_next()

test_it = test_dataset.make_one_shot_iterator()
test_next = test_it.get_next()

# build LSTM
weights = {
    'out': tf.Variable(tf.random_normal([NUM_UNITS, NUM_CLASSES]))
}
biases = {
    'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
}

X = tf.placeholder("float", [None, LENGTH, N_INPUT])
Y = tf.placeholder("float", [None, NUM_CLASSES])

def network(x, weights, biases):
    inp = tf.unstack(x, LENGTH, axis=1)
    network = tf.contrib.rnn.BasicLSTMCell(NUM_UNITS)
    output, states = rnn.static_rnn(network, inp, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
opt = tf.train.GradientDescentOptimizer(learning_rate = RATE)
train = op.minimize(loss)

correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# training
with tf.Session() as sess:
    sess.run(init)

    for step in range(EPOCH):
        batch_x, batch_y = train_next
        sess.run(train, feed_dict={X: batch_x, Y: batch_y})
        if step % 100 == 0:
            l, a = sess.run([loss, acc], feed_dict={X:batch_x, Y: batch_y})
            print("Step" + str(step) + ", Loss= " + \
                    "{:.3f}".format(l) + ", Acc= " + \
                    "{:.3f}".format(a))
    
    print("Test Acc:", sess.run(acc, feed_dict={X: test, Y: test_label}))



