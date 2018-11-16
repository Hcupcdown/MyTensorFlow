#  "first tensorflow experiment"
#create by cup——cdown
#2018-11-15

#引入tensorflow库
import tensorflow as tf
#载入训练集MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#引入pylab库
import pylab

#新建图
tf.reset_default_graph()
#定义占位符
x=tf.placeholder(tf.float32,[None,784])           #MNIST数据集维度为28*28=784
y=tf.placeholder(tf.float32,[None,10])            #共10个类别，从0~9

#构建模型
w=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.zeros([10]))

#softmax分类,损失函数
pred=tf.nn.softmax(tf.matmul(x,w)+b)
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

#定义学习率,使用梯度下降优化器
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#训练循环25次，每次训练100个数据，每一步显示一次训练结果
training_epochs=50
batch_size=100
display_step=1
saver=tf.train.Saver()
model_path="log/firstmodel.ckpt"
#启动会话
with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost+=c/total_batch
        if (epoch+1)%display_step==0:
            print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("Finished!")
    #测试模型
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #计算准确率
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    #保存模型
    save_path=saver.save(sess,model_path)
    print("Mode saved in file: %s" %save_path)
