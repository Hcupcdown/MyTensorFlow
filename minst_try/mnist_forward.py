#creatd by cup&cdown
#2019-2-26
#定义前向传播节点
import tensorflow as tf

#输入输出节点个数
INPUT_NODE = 784
OUTPUT_NODE = 10
#第一隐藏层节点个数
LAYER1_NODE = 500


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


def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE ],regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE , OUTPUT_NODE],regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y