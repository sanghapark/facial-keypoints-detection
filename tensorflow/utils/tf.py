import tensorflow as tf


def load_ckpt(checkpoint):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, checkpoint)
        print("Model restored.")

        # Check the values of the variables
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())