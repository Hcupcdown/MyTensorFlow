#creatd by cup&cdown
#2019-2-26
#测试训练准确度
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_CNN_backward
import mnist_CNN_forward
import numpy as np
TEST_INTERVAL_SECS = 5

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [mnist.test.num_examples,
         mnist_CNN_forward.INPUT_SIZE,
         mnist_CNN_forward.INPUT_SIZE,
         mnist_CNN_forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, mnist_CNN_forward.OUTPUT_NODE])
        y = mnist_CNN_forward.forward(x,False,None)

        ema = tf.train.ExponentialMovingAverage(mnist_CNN_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_CNN_backward.MODE_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshape_xs = np.reshape(mnist.test.images, (mnist.test.num_examples,mnist_CNN_forward.INPUT_SIZE,mnist_CNN_forward.INPUT_SIZE,mnist_CNN_forward.NUM_CHANNELS))
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshape_xs, y_: mnist.test.labels})
                    print("After %s training steps, test accuracy = %g" %(global_step,accuracy_score))
                else:
                    print("No checkpoint file found") 
                    return 
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./data/",one_hot=True)
    test(mnist)

if __name__ =='__main__':
    main()