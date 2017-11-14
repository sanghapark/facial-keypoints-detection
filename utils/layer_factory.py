import tensorflow as tf

def create_convolutional_layer(layer_count,
                               input_mat,
                               conv2_filter_size,
                               conv2_filter_stride,
                               conv2_padding,
                               maxpool_filter_size,
                               maxpool_filter_stride,
                               maxpool_padding,
                               dropout_rate,
                               batch_size,
                               color_channel,
                               conv2_depth):
    with tf.variable_scope('Conv2DLayer{:02d}'.format(layer_count)) as scope:
        print('Conv2DLayer{:02d}'.format(layer_count))
        print("Input: ", input_mat.shape)
        # conv2d
        conv2_size = [conv2_filter_size, conv2_filter_size, int(input_mat.shape[3]), conv2_depth]
        conv2_strides = [batch_size, conv2_filter_stride, conv2_filter_stride, color_channel]
        W = tf.Variable(tf.random_normal(conv2_size, stddev=0.01), name='W{:02d}'.format(layer_count))
        L = tf.nn.conv2d(input_mat, W, strides=conv2_strides, padding=conv2_padding)
        print("After Conv2: ", L.shape)

        # relu activation
        L = tf.nn.relu(L)

        mp_ksize = [batch_size, maxpool_filter_size, maxpool_filter_size, color_channel]
        mp_strides = [batch_size, maxpool_filter_stride, maxpool_filter_stride, color_channel]
        L = tf.nn.max_pool(L, ksize=mp_ksize, strides=mp_strides, padding=maxpool_padding)
        print("After Max Pooling: ", L.shape)

        # drop out regularization
        L = tf.nn.dropout(L, keep_prob=dropout_rate)

        print("="*50)
        return L

def create_dense_layer(layer_count,
                       input_vector,
                       flat_depth,
                       dropout_rate):
    with tf.variable_scope('DenseLayer{:02d}'.format(layer_count)) as scope:
        print('DenseLayer{:02d}'.format(layer_count))
        print("Input: ", input_vector.shape)
        shape = [input_vector.shape[1], flat_depth]
        init = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable('W{:02d}'.format(layer_count), shape=shape, initializer=init)
        bias = tf.Variable(tf.random_normal([flat_depth]))
        L = tf.nn.relu(tf.matmul(input_vector, W) + bias)
        L = tf.nn.dropout(L, keep_prob=dropout_rate)
        print("L.shape: ", L.shape)
        print("="*50)
        return L

def create_last_layer(layer_count,
                      input_vector,
                      flat_depth):
    with tf.variable_scope('DenseLayer{:02d}'.format(layer_count)) as scope:
        print('DenseLayer{:02d}'.format(layer_count))
        print("Input: ", input_vector.shape)
        shape = [input_vector.shape[1], flat_depth]
        init = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable('W{:02d}'.format(layer_count), shape=shape, initializer=init)
        bias = tf.Variable(tf.random_normal([flat_depth]))
        hypothesis = tf.matmul(input_vector, W) + bias
        print("hypothesis.shape: ", hypothesis.shape)
        print("="*50)
        return hypothesis
