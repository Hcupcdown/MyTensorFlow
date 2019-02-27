#反向传播模块
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pylab as plt
import forward
import generates
STEPS = 4000
BATCH_SIZE = 30
LEARING_RATE_BASE =0.001
LEARING_RATE_DECAY = 0.999
REGULARIZER = 0.01

def backword():
    x=tf.placeholder( tf.float32 , shape = (None,2))
    y_ = tf.placeholder(tf.float32, shape= (None,1))
    
    X, Y_, Y_c=generates.generateds()
    y=forward.forward(x,REGULARIZER)
    global_step=tf.Variable(0,trainable=False)
    learing_rate=tf.train.exponential_decay(
        LEARING_RATE_BASE,
        global_step,
        300/BATCH_SIZE,
        LEARING_RATE_DECAY,
        staircase=True
    )

    #定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    #定义反向传播算法
    train_step = tf.train.AdamOptimizer(learing_rate).minimize(loss_total)

    with tf.Session as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start=(i*BATCH_SIZE)%300
            end = start+BATCH_SIZE
            sess.run(train_step,feed_dict={x:X[start:end],y:Y_[start,end]})
            if i% 2000 ==0:
                loss_v =sess.run(loss_total,feed_dict={x:X,y:Y_})
                print("After %d steps ,loss is :%f"%(i,loss_v))
        xx,yy =np.mgrid[-3:3:.01,-3:3:.01]
        grid =np.c_[xx.ravel(),yy.ravel()]
        probs=sess.run(y, feed_dict={x:grid})
        probs=probs.reshape(xx.shape)
    plt.scatter(X[:,0],X[:1],c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()