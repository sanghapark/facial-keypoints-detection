import numpy as np
import tensorflow as tf
from utils.constants import *
from utils.layer_factory import *


# Hyperparameters
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.95
# learning_rate = 0.001

conv2_filter_size = 2
conv2_filter_stride = 1
conv2_padding = "VALID"
conv2_starting_depth = 20
conv2_depth_exp_multiple = 3

maxpool_decline = 12
maxpool_filter_size = maxpool_decline+1
maxpool_filter_stride = 1
maxpool_padding = "VALID"

flat_starting_depth = 500
flat_decay_rate = 0.8

cnn_batch_size = 1
dropout_rate = 0.5


class CnnModel:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout rate
            # 0 < dropoutrate < 1.0 for training
            # 1 for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input placeholders
            self.X = tf.placeholder(tf.float32, [None, img_width*img_height])

            # img 96x96x1 (greyscale)
            X_img = tf.reshape(self.X, [-1, img_width, img_width, 1])
            self.Y = tf.placeholder(tf.float32, [None, n_features])

            # 이미지데이터 정보 = (?, 96, 96, 1)
            # (batch_size, width, height, 1 for greyscale, 3 for RGB)
            # Filter Size = [3, 3, 1, 100] 
            # 100개의 3x3x1 filters
            # Maybe initi with Xavier??
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 100], stddev=0.01))

            # Conv 사이즈는 이미지 사이즈와 같게 만들어 보자 (?, 96, 96, 32)
            # stride 양옆 위아래고 두개씩 더해지므로 96 -> 96 + 2
            # (N-F)/stride + 1 = (96+2-3)/1 + 1 = 96
            # padding: SAME-> with zero padding, VALID-> without padding
            # outpuy size = (?, 96, 96, 100)
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            
            # relu를 사용한 activation function
            # 다른 옵션들은 LeakyReLU, Maxout, ELU
            # output size = (?, 96, 96, 100)
            L1 = tf.nn.relu(L1)

            # Pooling (Sampling)
            # 2x2사이즈의 필터링으로 Maximum Sampling 사용
            # output size: (?, 48, 48, 1) 
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Dropout
            # output size: (?, 48, 48, 1) 
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
            '''
            Tensor("Conv2D:0", shape=(?, 96, 96, 100), dtype=float32)
            Tensor("Relu:0", shape=(?, 96, 96, 100), dtype=float32)
            Tensor("MaxPool:0", shape=(?, 48, 48, 100), dtype=float32)
            Tensor("dropout/mul:0", shape=(?, 48, 48, 100), dtype=float32)
            '''

            # L2 ImgIn shape=(?, 48, 48, 100)
            W2 = tf.Variable(tf.random_normal([3, 3, 100, 200], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            '''
            Tensor("Conv2D_1:0", shape=(?, 48, 48, 200), dtype=float32)
            Tensor("Relu_1:0", shape=(?, 48, 48, 200), dtype=float32)
            Tensor("MaxPool_1:0", shape=(?, 24, 24, 200), dtype=float32)
            Tensor("dropout_1/mul:0", shape=(?, 24, 24, 200), dtype=float32)
            '''

            # L3 ImgIn shape=(?, 24, 24, 200)
            W3 = tf.Variable(tf.random_normal([3, 3, 200, 400], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            L3_flat = tf.reshape(L3, [-1, 400*12*12])
            '''
            Tensor("Conv2D_2:0", shape=(?, 24, 24, 400), dtype=float32)
            Tensor("Relu_2:0", shape=(?, 24, 24, 400), dtype=float32)
            Tensor("MaxPool_2:0", shape=(?, 12, 12, 400), dtype=float32)
            Tensor("dropout_2/mul:0", shape=(?, 12, 12, 400), dtype=float32)
            Tensor("Reshape_1:0", shape=(?, 57600), dtype=float32)
            '''

            # L4 Fully Connected 12x12x400 -> 1000 outputs
            W4 = tf.get_variable("W4", shape=[400*12*12, 1000], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([1000]))
            L4_flat = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
            L4_flat = tf.nn.dropout(L4_flat, keep_prob=self.keep_prob)
            '''
            Tensor("Relu_3:0", shape=(?, 1000), dtype=float32)
            Tensor("dropout_3/mul:0", shape=(?, 1000), dtype=float32)
            '''

            # L5 FC 1000 inputs -> 500 outputs
            W5 = tf.get_variable("W5", shape=[1000, 500], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([500]))
            L5_flat = tf.nn.relu(tf.matmul(L4_flat, W5) + b5)
            L5_flat = tf.nn.dropout(L5_flat, keep_prob=self.keep_prob)
            '''
            Tensor("Relu_4:0", shape=(?, 500), dtype=float32)
            Tensor("dropout_4/mul:0", shape=(?, 500), dtype=float32)
            '''
            
            # L6 Final FC 500 inputs -> 30 outputs
            W6 = tf.get_variable("W6", shape=[500, 30], initializer=tf.contrib.layers.xavier_initializer())
            b6 = tf.Variable(tf.random_normal([30]))
            '''
            Tensor("add_1:0", shape=(?, 30), dtype=float32)
            '''

            self.hypothesis = tf.matmul(L5_flat, W6) + b6

        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step*BATCH_SIZE, decay_steps, decay_rate)
  
        # define cost/loss & optimizer
        # self.cost = tf.reduce_mean(tf.square(tf.subtract(self.hypothesis, self.Y)), name="cost")
        self.cost = tf.reduce_sum(tf.squared_difference(self.hypothesis, self.Y), name="cost")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.summary = tf.summary.merge_all()

        tf.summary.scalar("loss", self.cost)


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