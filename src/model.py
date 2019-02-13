import numpy as np 
import os
from PIL import Image # gives better output control than matplotlib
import classes
import tensorflow as tf
import random
import datetime
import sklearn.model_selection as sk
from sklearn.preprocessing import StandardScaler
from operator import mul
import copy
import pandas as pd

class Model:
	def __init__(self, im_shape, n_classes):
		self.x_dim, self.y_dim, self.n_channels = im_shape
		self.n_classes = n_classes
		# self.inp = None
		# self.out = None
		# self.drop_rate = None

	def basic_cnn(self):
		conv1_filters = 16
		conv2_filters = 32
		dense_size = 1024  
	 
		self.inp = tf.placeholder(tf.float32, [None, self.x_dim,self.y_dim,self.n_channels], name = "input")
		drop_rate = tf.placeholder(tf.float32)


		self.out = tf.placeholder(tf.int32, [None], name = "output")

		# Convolutional Layer #1
		conv1 = tf.layers.conv2d(
			inputs=self.inp,
			filters=conv1_filters,
			data_format = 'channels_last',
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu)

		# Pooling Layer #1
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

		# Convolutional Layer #2 and Pooling Layer #2
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=conv2_filters,
			kernel_size=[5, 5],
			padding="same",
			data_format = 'channels_last',
			activation=tf.nn.relu)
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
		pool2_flat =tf.reshape(pool2, [-1, int(self.x_dim/4) * int(self.y_dim/4) * conv2_filters])
		dense = tf.layers.dense(inputs=pool2_flat, units=dense_size, activation=tf.nn.relu)
		dropout = tf.layers.dropout(
			inputs=dense, rate=0.4 )
		self.logits = tf.layers.dense(inputs=dropout, units = self.n_classes)

	def opt(self):

		learning_rate = 0.01
		self.cost = tf.losses.sparse_softmax_cross_entropy(labels=self.out, logits=self.logits)
		self.optimiser = tf.train.AdamOptimizer( 0.01 ).minimize(self.cost)