#firstCNN
#第一个卷积神经网络demo
#create by cup&cdown 
#2018-11-16
import tensorflow as tf
import matplotlib.pyplot as plt         #plt用于显示图片
import matplotlib.image as mpimg        #mpimg用于读取图片
import numpy as np 

myimg=mpimg.imread('D:/资源文件/Python/firstCNN/img.jpg')
plt.imshow(myimg)
plt.axis('off')
plt.show()
print(myimg.shape)

#重新设定输入图片的形状
full=np.reshape(myimg,[1,735,431,3])
#输入的变量
inputfull=tf.Variable(tf.constant(1.0,shape=[1,735,431,3]))
#过滤器
#shape中参数为：卷积核高度，宽度，图像通道数，卷积核个数
filter=tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0.0,0.0,0.0],[1.0,1.0,1.0],[-2.0,-2.0,-2.0],[0,0,0],[2.0,2.0,2.0],[-1.0,-1.0,-1.0],[0.0,0.0,0.0],[1.0,1.0,1.0]],shape=[3,3,3,1]))
#输入图像，过滤器，，填充为SAME表示为同卷积操作
op=tf.nn.conv2d(inputfull,filter,strides=[1,1,1,1],padding='SAME')
#tf.cast用于改变数据类型
o=tf.cast(((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_max(op)-tf.reduce_min(op)))*255,tf.uint8)

with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    #执行o和filter操作
    t,f=sess.run([o,filter],feed_dict={inputfull:full})
    #更改t的形状为[82,83]
    t=np.reshape(t,[735,431])
    plt.imshow(t,cmap="Greys_r")
    plt.axis('off')
    plt.show()
