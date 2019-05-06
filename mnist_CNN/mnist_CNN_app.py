#created by cup&cdown
#2018-12-26
#大创项目代码

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_CNN_forward
import mnist_CNN_backward
import matplotlib.pylab as plt

def restore_model(testPicArr):
    #生成计算图，并将计算图设为默认
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [1, mnist_CNN_forward.INPUT_SIZE,mnist_CNN_forward.INPUT_SIZE,mnist_CNN_forward.NUM_CHANNELS])
        y = mnist_CNN_forward.forward(x,False, None)
        preValue = tf.argmax(y, 1)
        
        variable_averages = tf.train.ExponentialMovingAverage(mnist_CNN_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        testPicArr = np.reshape(testPicArr,(1,28,28,1))
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_CNN_backward.MODE_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                preValue = sess.run(preValue, feed_dict={x:testPicArr})

                return preValue
            else:
                print("No checkpoint file found")
                return -1
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28,28), Image.ANTIALIAS)
    #将图片变为灰度图，并转换为array
    im_arr = np.array(reIm.convert('L'))
    #判断像素点为白或黑的阈值
    threshold = 120
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            #将图片转换为只包含黑白两个像素的图片
            if (im_arr[i][j]<threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    #plt.imshow(im_arr)
    #plt.show()
    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready

def application():
    #testNum = input("input the number of test pictures:")
    for i in range(3):
        testPic = input("the path  of test picture:")
        testPicArr = pre_pic(testPic)
        print(testPicArr)
        preValue = restore_model(testPicArr)
        print ("The prediction number is:" , preValue)

def main():
    application()

if __name__ == '__main__':
    main()