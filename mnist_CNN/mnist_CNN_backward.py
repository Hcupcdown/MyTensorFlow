#coding: UTF-8
#creatd by cup&cdown
#2019-2-26
#Lenet5反向传播训练
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_CNN_forward
import os
import numpy as np

BATCH_SIZE = 100
LENARNING_RATE_BASE = 0.005
LENARNING_RATE_DECAY= 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODE_SAVE_PATH = "./mode"
MODE_NAME="mnist_model"

def backward(mnist):

    x=tf.placeholder(tf.float32,[BATCH_SIZE, mnist_CNN_forward.INPUT_SIZE,mnist_CNN_forward.INPUT_SIZE,mnist_CNN_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32,[None, mnist_CNN_forward.OUTPUT_NODE])
    y = mnist_CNN_forward.forward(x,True,REGULARIZER)
    global_step = tf.Variable(0,trainable=False)
    
    #损失函数，交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cem =tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    
    #指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LENARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LENARNING_RATE_DECAY,
        staircase=True
    )

    #滑动平均优化,MOVING_AVERAGE_DECAY为平均衰减率
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    
    #trainable_variables将待训练参数汇总为列表
    #ema.apply 将对括号内所有参数求滑动平均
    ema_op=ema.apply(tf.trainable_variables())

    #control_dependencies将train_step和ema_op绑定，使两者同步运行
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    #保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #实现断点续训
        have_done = 0
        ckpt = tf.train.get_checkpoint_state(MODE_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #从名字中分割出上次训练到的次数
            have_done = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

        for i in range(have_done,STEPS):
            xs, ys =mnist.train.next_batch(BATCH_SIZE)
            #将输入的mnist数据集整形为需要的形状
            reshape_xs = np.reshape(xs,(
                BATCH_SIZE,
                mnist_CNN_forward.INPUT_SIZE,
                mnist_CNN_forward.INPUT_SIZE,
                mnist_CNN_forward.NUM_CHANNELS))
            _, loss_value,step =sess.run([train_op, loss, global_step], feed_dict={x: reshape_xs, y_: ys})
            if i % 1000 ==0:
                print("after %d training steps, loss on traning batch is %g" % (step, loss_value))
                saver.save(sess,os.path.join(MODE_SAVE_PATH,MODE_NAME), global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/",one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()