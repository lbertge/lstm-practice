import tensorflow as tf
import numpy as np

NUM_UNITS = 500
NUM_UNITS = 500
LENGTH = 10
TOTAL_SIZE = 1000
DATA_SIZE = int(0.6 * TOTAL_SIZE)
VAL_SIZE = int(0.2 * TOTAL_SIZE)
TEST_SIZE = int(0.2 * TOTAL_SIZE)
LR = 0.001
EPSILON = 1e-8
ALPHA = 0.0001

EPOCH = 10000
PRINT_EVERY_EPOCH = EPOCH / 10

x_ = tf.placeholder(tf.float32, shape=[None,LENGTH])
y_ = tf.placeholder(tf.float32, shape=[None,1])

t1 = tf.Variable(tf.random_uniform([LENGTH, NUM_UNITS], -1, 1))
t2 = tf.Variable(tf.random_uniform([NUM_UNITS, NUM_UNITS], -1, 1))
t3 = tf.Variable(tf.random_uniform([NUM_UNITS, 1], -1, 1))

b1 = tf.Variable(tf.zeros([NUM_UNITS]))
b2 = tf.Variable(tf.zeros([NUM_UNITS]))
b3 = tf.Variable(tf.zeros([1]))

act1 = tf.tanh(tf.matmul(x_, t1) + b1)
act2 = tf.tanh(tf.matmul(act1, t2) + b2)
out = tf.sigmoid(tf.matmul(act2, t3) + b3)

pred = tf.round(out)
correct = tf.equal(pred, y_)
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

cost = tf.reduce_mean( -1 * (y_ * tf.log(out + EPSILON) + (1 - y_) * tf.log((1.0 - out) + EPSILON))) # cross ent

train = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-08).minimize(cost)

#test_X = tf.floor(tf.random_uniform([DATA_SIZE, LENGTH]) * 2)
#test_Y = tf.mod(tf.reduce_sum(test_X, 1), tf.constant(2.0))

test_X = np.floor(np.random.rand(TEST_SIZE, LENGTH) * 2)
test_Y = np.mod(np.sum(test_X, 1), 2.0)
test_Y = test_Y.reshape(-1, 1)

valid_X = np.floor(np.random.rand(VAL_SIZE, LENGTH) * 2)
valid_Y = np.mod(np.sum(valid_X, 1), 2.0)
valid_Y = valid_Y.reshape(-1, 1)

train_X = np.floor(np.random.rand(DATA_SIZE, LENGTH) * 2)
train_Y = np.mod(np.sum(train_X, 1), 2.0)
train_Y = train_Y.reshape(-1, 1)

print(train_X, train_X.shape)
print(train_Y, train_Y.shape)

init = tf.global_variables_initializer()
min_valid_loss = TOTAL_SIZE
with tf.Session() as sess:
    sess.run(init)
    for i in range(EPOCH):
        train_loss = sess.run(train, feed_dict={x_: train_X, y_: train_Y})
        valid_loss, valid_acc = sess.run([cost, acc], feed_dict={x_: valid_X, y_: valid_Y})
        min_valid_loss = min(valid_loss, min_valid_loss)

        np.random.shuffle(train_X)
        train_Y = np.mod(np.sum(train_X, 1), 2.0)
        train_Y = train_Y.reshape(-1, 1)


        generalization_loss = 100 * (valid_loss / min_valid_loss - 1)
        #if generalization_loss > ALPHA:
        #    print('should stop now at epoch', i)

        if i % PRINT_EVERY_EPOCH == 0:
            print ('Epoch', i)
            print('valid acc: ', valid_acc)
            print('valid loss: ', valid_loss)
            print('generalization loss: ', generalization_loss)

    co, ac, p, cor = (sess.run([cost, acc, pred, correct], feed_dict={x_: test_X, y_: test_Y}))
    print(test_X[:4], test_Y[:4])

