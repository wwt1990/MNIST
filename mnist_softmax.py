import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b  # predicted labels

# define loss(cross_entropy) and optimizer(gradient descent)
y_ = tf.placeholder(tf.float32, [None, 10]) # true labels
    # raw formulation of cross_entropy:
    # sum_cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices [1])
sum_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
cross_entropy = tf.reduce_mean(sum_cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

# test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
