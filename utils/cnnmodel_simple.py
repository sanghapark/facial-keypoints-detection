import numpy as np
import tensorflow as tf
from utils.constants import *
from utils.layer_factory import *


# Hyperparameters
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.95
# learning_rate = 0.001


class CnnModel:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):

            self.keep_prob = tf.placeholder(tf.float32)
            self.X = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE])
            X_img = tf.reshape(self.X, [-1, IMG_SIZE, IMG_SIZE, 1])
            tf.summary.image('input', X_img, 3)
            self.Y = tf.placeholder(tf.float32, [None, N_KEYPOINTS])

            with tf.variable_scope('Conv2DLayer01') as scope:
                W1 = tf.Variable(tf.random_normal([3, 3, 1, 100], stddev=0.01))
                L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
                print("L1: ", L1.shape)

            with tf.variable_scope('Conv2DLayer02') as scope:
                W2 = tf.Variable(tf.random_normal([3, 3, 100, 200], stddev=0.01))
                L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
                L2 = tf.nn.relu(L2)
                L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
                L2_flat = tf.reshape(L2, [-1, 400*24*24])
                print("L2: ", L2.shape)

            with tf.variable_scope('DenseLayer01') as scope:
                # L4 Fully Connected 12x12x400 -> 1000 outputs
                W3 = tf.get_variable("W3", shape=[400*24*24, 1000], initializer=tf.contrib.layers.xavier_initializer())
                b3 = tf.Variable(tf.random_normal([1000]))
                L3_flat = tf.nn.relu(tf.matmul(L2_flat, W3) + b3)
                L3_flat = tf.nn.dropout(L3_flat, keep_prob=self.keep_prob)
                print("L3_flat: ", L3_flat.shape)

            with tf.variable_scope('DenseLayer02') as scope:
                W4 = tf.get_variable("W4", shape=[1000, 500], initializer=tf.contrib.layers.xavier_initializer())
                b4 = tf.Variable(tf.random_normal([500]))
                L4_flat = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
                L4_flat = tf.nn.dropout(L4_flat, keep_prob=self.keep_prob)
                print("L4_flat: ", L4_flat.shape)

            with tf.variable_scope('LastDenseLayer03') as scope:
                W5 = tf.get_variable("W5", shape=[500, 30], initializer=tf.contrib.layers.xavier_initializer())
                b5 = tf.Variable(tf.random_normal([30]))
                print("W5: ", W5.shape, "b5: ", b5.shape)

            self.hypothesis = tf.matmul(L4_flat, W5) + b5

        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step*BATCH_SIZE, decay_steps, decay_rate)
  
        self.cost = tf.reduce_sum(tf.squared_difference(self.hypothesis, self.Y), name="cost")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


    def predict(self, x_test, keep_prop=1.0):
        feed_dict = {
            self.X: x_test,
            self.keep_prob: keep_prop
        }
        return self.sess.run(self.hypothesis, feed_dict=feed_dict)
    
    def validate(self, X, Y, keep_prop=1.0):
        feed_dict = {
            self.X: X,
            self.Y: Y,
            self.keep_prob: keep_prop
        }
        return self.sess.run(self.cost, feed_dict=feed_dict)

    def train(self, x_data, y_data, keep_prop=0.5):
        feed_dict = {
            self.X: x_data,
            self.Y: y_data,
            self.keep_prob: keep_prop
        }
        return self.sess.run([self.cost, self.optimizer], feed_dict=feed_dict)

    def summarize(self, X, Y, keep_prop=0.5):
        feed_dict = {
            self.X: X,
            self.Y: Y,
            self.keep_prob: keep_prop
        }
        return self.sess.run(self.summary, feed_dict=feed_dict)