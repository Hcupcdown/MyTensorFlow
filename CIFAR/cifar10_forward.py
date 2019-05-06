#created by Hcupcdown
#2019-3-10
#CIFAR

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform
from six.moves import urllib
import tensorflow as tf
import cifar10_input

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLE_TRAIN = cifar10_input.NUM_EXAMPLE_TRAIN
NUM_EXAMPLE_EVAL = cifar10_input.NUM_EXAMPLE_TRAIN

#滑动平均
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LENARING_RATE_DECAY_FACTOR = 0.1
INITIAL_LENRING_RATE = 0.1

TOWER_NAME = 'tower'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name+ '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))