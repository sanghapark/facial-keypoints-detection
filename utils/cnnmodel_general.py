import numpy as np
import tensorflow as tf
from utils.constants import *
from utils.layer_factory import *


# Hyperparameters
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.95

conv2_filter_size = 2
conv2_filter_stride = 1
conv2_padding = "VALID"
conv2_starting_depth = 50
conv2_depth_exp_multiple = 3

maxpool_decline = 8
maxpool_filter_size = maxpool_decline+1
maxpool_filter_stride = 1
maxpool_padding = "VALID"

flat_starting_depth = 1000
flat_decay_rate = 0.7

cnn_batch_size = 1
dropout_rate = 0.5
depth_multiple = 1 + (maxpool_filter_size/IMG_SIZE)


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
            self.Y = tf.placeholder(tf.float32, [None, N_KEYPOINTS])

            L = X_img
            layer_count = 0
            conv2_depth = conv2_starting_depth
            while True:
                L = create_convolutional_layer(layer_count+1,
                                               L,
                                               conv2_filter_size,
                                               conv2_filter_stride,
                                               conv2_padding,
                                               maxpool_filter_size,
                                               maxpool_filter_stride,
                                               maxpool_padding,
                                               dropout_rate,
                                               cnn_batch_size,
                                               N_CHANNELS,
                                               conv2_depth)
                layer_count += 1
                if L.shape[1] <= maxpool_decline:
                    break
                conv2_depth = int(np.ceil(int(L.shape[3])*(1+(maxpool_decline/IMG_SIZE))**conv2_depth_exp_multiple))


            L_flat = tf.reshape(L, [-1, int(L.shape[1] * L.shape[2] * L.shape[3])])

            layer_count += 1
            flat_depth = int(L_flat.shape[1])
            while True:
                L_flat = create_dense_layer(layer_count, L_flat, flat_depth, dropout_rate)
                layer_count += 1
                flat_depth = int(int(L_flat.shape[1]) ** flat_decay_rate)
                if flat_depth <= N_KEYPOINTS:
                    break

            self.hypothesis = create_last_layer(layer_count, L_flat, N_KEYPOINTS)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step*BATCH_SIZE, decay_steps, decay_rate)
  
        # define cost/loss & optimizer
        # self.cost = tf.reduce_mean(tf.square(tf.subtract(self.hypothesis, self.Y)), name="cost")
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.hypothesis, self.Y))), name="cost")
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