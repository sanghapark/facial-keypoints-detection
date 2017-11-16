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

    def _create_conv2d(idx, input, filters, kernel_size, conv_padding, conv2_strides, activation, regularizers, pool_size, pool_padding, pool_strides):
        with tf.variable_scope('conv2d{}'.format(idx)) as scope:
            conv = tf.layers.conv2d(inputs=X_img, filters=filters, kernel_size=kernel_size, padding=conv_padding, activation=activation)
            pool = tf.layers.max_pooling2d(inputs=conv, pool_size=pool_size, padding=pool_padding, strides=pool_strides)
            dropout = tf.layers.dropout(inputs=pool, rate=self.keep_prob, training=self.training)
            print(conv.shape, pool.shape, dropout.shape)


    def _build_net(self):
        with tf.variable_scope(self.name):

            self.training = tf.placeholder(tf.bool)

            self.keep_prob = tf.placeholder(tf.float32)
            self.X = tf.placeholder(tf.float32, [None, 96*96])
            X_img = tf.reshape(self.X, [-1, 96, 96, 1])
            tf.summary.image('input', X_img, 3)
            self.Y = tf.placeholder(tf.float32, [None, 30])

            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            learning_rate = tf.train.exponential_decay(initial_learning_rate, self.global_step*BATCH_SIZE, decay_steps, decay_rate)

            reg_l1 = tf.contrib.layers.l1_regularizer(scale=0.01)
            reg_l2 = tf.contrib.layers.l2_regularizer(scale=0.01)
            reg = None

            with tf.variable_scope('conv2d01') as scope:
                conv1 = tf.layers.conv2d(inputs=X_img, 
                                        filters=100, 
                                        kernel_size=[3, 3], 
                                        padding='SAME', 
                                        activation=tf.nn.elu,
                                        # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=reg)
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], padding='VALID', strides=2)
                dropout1 = tf.layers.dropout(inputs=pool1, rate=self.keep_prob, training=self.training)
                print(conv1.shape, pool1.shape, dropout1.shape)

            with tf.variable_scope('conv2d02') as scope:
                conv2 = tf.layers.conv2d(inputs=dropout1, 
                                        filters=200, 
                                        kernel_size=[3, 3], 
                                        padding='SAME', 
                                        activation=tf.nn.elu, 
                                        # kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                        kernel_regularizer=reg)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], padding='VALID', strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2, rate=self.keep_prob, training=self.training)
                print(conv2.shape, pool2.shape, dropout2.shape)

            with tf.variable_scope('conv2d03') as scope:
                conv3 = tf.layers.conv2d(inputs=dropout2,
                                        filters=400,
                                        kernel_size=[2, 2],
                                        padding='SAME',
                                        activation=tf.nn.elu,
                                        # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=reg)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding='VALID', strides=1)
                dropout3 = tf.layers.dropout(inputs=pool3, rate=self.keep_prob, training=self.training)
                print(conv3.shape, pool3.shape, dropout3.shape)

            with tf.variable_scope('dense04') as scope:
                flat1 = tf.reshape(dropout3, [-1, 400*22*22])
                dense4 = tf.layers.dense(inputs=flat1,
                                        units=1000,
                                        activation=tf.nn.elu,
                                        # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=reg)
                dropout4 = tf.layers.dropout(inputs=dense4, rate=self.keep_prob, training=self.training)

            with tf.variable_scope('dense05') as scope:
                self.hypothesis = tf.layers.dense(inputs=dropout4,
                                                units=30, activation=None,
                                                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                kernel_regularizer=reg)
                print("hypothesis shape: ", self.hypothesis.shape)

        self.cost = tf.reduce_sum(tf.squared_difference(self.hypothesis, self.Y), name="cost")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


    def predict(self, x_test, training=False, keep_prop=1.0):
        feed_dict = {
            self.X: x_test,
            self.keep_prob: keep_prop,
            self.training: training
        }
        return self.sess.run(self.hypothesis, feed_dict=feed_dict)
    
    def validate(self, X, Y, training=False, keep_prop=1.0):
        feed_dict = {
            self.X: X,
            self.Y: Y,
            self.keep_prob: keep_prop,
            self.training: training
        }
        return self.sess.run(self.cost, feed_dict=feed_dict)

    def train(self, x_data, y_data, global_step, training=True, keep_prop=0.5):
        feed_dict = {
            self.X: x_data,
            self.Y: y_data,
            self.global_step: global_step,
            self.keep_prob: keep_prop,
            self.training: training
        }
        return self.sess.run([self.cost, self.optimizer], feed_dict=feed_dict)

    def summarize(self, X, Y, keep_prop=0.5):
        feed_dict = {
            self.X: X,
            self.Y: Y,
            self.keep_prob: keep_prop
        }
        return self.sess.run(self.summary, feed_dict=feed_dict)