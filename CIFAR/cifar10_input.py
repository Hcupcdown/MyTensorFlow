#created by Hcupcdown
#2019-3-10 
#CIFAR

#为了在老版本中兼容新特性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform
from six.moves import xrange
import tensorflow as tf

from tensorflow.python.platform import gfile

IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLE_TRAIN = 50000
NUM_EXAMPLE_EVAL = 10000

def read_cifar10(filename_queue):
    class CIFARRecord(object):
        pass
    result = CIFARRecord()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.channel = 3
    image_bytes = result.height * result.width * result.channel

    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(tf.slice(record_bytes, [0],[label_bytes]),tf.int32)

    depth_major = tf.reshape(tf.slice(record_bytes,[label_bytes],[image_bytes]),
                            [result.channel,result.height,result.width])
    #将channel，height，width格式转换为height,width,channel [1,2,0]将原来0列放到第2列
    result.uint8image = tf.transpose(depth_major,[1,2,0])

    return result

def _generate_image_and_label_bacth(image, label, min_queue_examples,batch_size):
    num_preprocess_threads = 16
    #随机洗牌队列
    images, label_batch = tf.train.shuffle_batch(
        [image,label],
        batch_size=batch_size,
        #排队线程数
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3*batch_size,
        min_after_dequeue=min_queue_examples
    )
    tf.image_summary('images',images)

    return images, tf.reshape(label_batch, [batch_size])

def distorted_input(data_dir, batch_size):
    #路径拼接
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %i)
                    for i in xrange(1,6)]
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    
    filename_queue = tf.train.string_input_producer(filenaems)

    read_input = read_cifar10(filename_queue)
    reshape_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distorted_image = tf.image.random_crop(reshape_image,[height,width])

    distorted_image = tf.image.random_flip_left_right(distorted_image)

    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)

    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)

    float_image = tf.image.per_image_whitening(distorted_image)

    min_fraction_of_example_in_queue = 0.4
    min_queue_example = int(NUM_EXAMPLE_TRAIN*min_fraction_of_example_in_queue)

    print('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_bacth(float_image, read_input.label,
                                            min_queue_examples,batch_size)

def input(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin'%i)
                                    for i in xrange(1,6)]
        num_example = NUM_EXAMPLE_TRAIN
    else:
        filenames = [os.path.join(data.dir, 'test_batch.bin')]
        num_example_ = NUM_EXAMPLE_EVAL

    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file:'+ f)
    filename_queue = tf.train.string_input_producer(filenames)

    read_input =read_cifar10(filename_queue)
    reshape_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resize_image = tf.image.resize_image_with_crop_or_pad(reshape_image,width, height)

    float_image = tf.image.per_image_whitening(resize_image)

    min_fraction_of_example_in_queue =0.4
    min_queue_examples = int(num_example*min_fraction_of_example_in_queue)

    return _generate_image_and_label_bacth(float_image, read_input.label,min_queue_examples,batch_size)