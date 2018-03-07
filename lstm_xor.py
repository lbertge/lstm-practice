from __future__ import print_function
import numpy as np
import tensorflow as tf
from numpy.random import shuffle
from tensorflow.contrib.rnn import LSTMCell
import matplotlib.pyplot as plt


def dataset(size):
    data = ["{0:012b}".format(i) for i in xrange(size)]
