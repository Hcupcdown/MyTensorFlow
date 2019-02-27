##coding: UTF-8
#creatd by cup&cdown
#2019-2-26
#Lenet5定义前向传播节点
import tensorflow as tf

#定义节点个数
INPUT_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
#第一层卷积核个数
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
#全连接层的节点个数
FC_SIZE = 512
OUTPUT_NODE = 10

def get_weight(shape, regularizer):
    #过滤掉偏离较大的正态分布
    w = tf.Variable(tf.truncated_normal(shape,stddev = 0.1))
    #是否正则化
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

#卷积操作
def conv2d(x, w):
    #conv2d参数为：输入描述（batch，行分辨率，列分辨率，通道数），卷积核描述，滑动步长
    return tf.nn.conv2d(x , w,strides=[1,1,1,1],padding="SAME")

#最大池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding = "SAME")

def forward(x, train,regularizer):
    #第一卷积层
    conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],regularizer)
    conv1_b =get_bias([CONV1_KERNEL_NUM])
    conv1=conv2d(x, conv1_w)
    #激活函数relu
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    pool1 = max_pool_2x2(relu1)

    #第二层卷积
    conv2_w = get_weight([CONV2_SIZE,CONV1_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    #将pool2的shape存为list
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshape = tf.reshape(pool2, [pool_shape[0],nodes])

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshape, fc1_w) + fc1_b)
    if train:
        #在训练过程中随机舍弃一部分参数，避免过拟合，加快训练速度
        fc1 = tf.nn.dropout(fc1, 0.5)
    
    fc2_w =get_weight([FC_SIZE, OUTPUT_NODE],regularizer)
    fc2_b = get_bias ([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y