#creatd by cup_cdown
#2018-11-26
#CNN Auto encoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#导入minst数据集中的input
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#定义学习率等参数
learning_rate=0.01
#第一层有256个节点，第二层有128个
n_hidden_1=256
n_hidden_2=128
n_input=784

#占位符
x=tf.placeholder("float",[None,n_input])
y=x

#学习参数w，b
weights={
    'encoder_conv1':tf.Variable(tf.truncated_normal([5,5,1,1],stddev=0.1)),
    'encoder_conv2':tf.Variable(tf.truncated_normal([3,3,1,1],stddev=0.1)),
    'decoder_conv1':tf.Variable(tf.truncated_normal([5,5,1,1],stddev=0.1)),
    'decoder_conv2':tf.Variable(tf.truncated_normal([3,3,1,1],stddev=0.1))
}

biases={
    'encoder_conv1':tf.Variable(tf.zeros([n_hidden_1])),
    'encoder_conv2':tf.Variable(tf.zeros([n_hidden_2])),
    'decoder_conv1':tf.Variable(tf.zeros([n_hidden_1])),
    'decoder_conv2':tf.Variable(tf.zeros([n_input]))
}

x_image=tf.reshape(x,[-1,28,28,1])

#编码
def encoder(x):
    h_conv1=tf.nn.relu(tf.nn.conv2d(x,weights['encoder_conv1'])+biases['encoder_conv1'])
    h_conv2=tf.nn.relu(tf.nn.conv2d(h_conv1,weights['encoder_conv2'])+biases['encoder_conv2'])
    return h_conv2,h_conv1

#解码
def decoder(x,conv1):
    t_conv1=tf.nn.conv2d_transpose(x-biases['decoder_conv2'],weights['decoder_conv2'],conv1.shape,[1,1,1,1])
    t_x_image=tf.nn.conv2d_transpose(t_conv1-biases['decoder_conv1'],weights['decoder_conv1'],x_image.shape,[1,1,1,1])
    return t_x_image

encoder_out,conv1=encoder(x_image)
h_pool2,mask=max_pool_with_argmax(encoder_out,2)

#反池化
h_upool=unpool(h_pool2,mask,2)
pred=decoder(h_upool,conv1)

#cost值为y与pred的平方差
cost=tf.reduce_mean(tf.pow(y-pred,2))
optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#定义学习率
training_epochs=20
batch_size=256
display_step=5

#开始模型训练
with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    total_batch=int(mnist.train.num_examples/batch_size)

    #开始训练
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs})
        if epoch % display_step==0:
            print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(c))
    print("finish!")

    #输出错误率
    batch_xs,batch_ys=mnist.train.next_batch(batch_size)
    print("Error:",cost.eval({x:batch_xs}))

    #输出可视化检测结果
    show_num=10
    reconstruction =sess.run(pred,feed_dict={x:batch_xs})
    f,a=plt.subplots(2,10,figsize=(10,2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(batch_xs[i],(28,28)))
        a[1][i].imshow(np.reshape(reconstruction[i],(28,28)))
    plt.draw()
