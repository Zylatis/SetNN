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
import fns

class CNN:
	def __init__(self, im_shape, n_classes, hyperpars, **kwargs):
		self.x_dim, self.y_dim, self.n_channels = im_shape
		self.n_classes = n_classes
		self.hyperpars = hyperpars
		print("Setup model:"),

		# Merge kwargs and hyperpars into a temporary dict to make object variables
		# (this may not be ideal down the line, though nothing is truly private in python anyway soooo...)
		local_defs = copy.deepcopy(hyperpars)
		local_defs.update(kwargs)

		# atm the hyperpars and kwargs are diff to allow feed_dict to do stuff
		for k,v in local_defs.items():
			try:
				assert k not in self.__dict__
			except AssertionError as e:
				print("\nWARNING: overriding variable '" + k + "' in model definition")

			try:
				setattr(self, k, v) # could use self.__dict__.update but ill advised apparently
			except Exception as e:
				print("\nCouldn't set " + k + " to value " + str(v) + " in model definition")
		print("Done")
	
	def build_layers(self):
		conv1_filters = 16
		conv2_filters = 32
		dense_size = 1024  

		# Placeholders for input/output (fed from feed_dict)	 
		self.inp = tf.placeholder(tf.float32, [None, self.x_dim,self.y_dim,self.n_channels], name = "input")
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
			inputs=dense, rate=self.drop_rate )
		self.logits = tf.layers.dense(inputs=dropout, units = self.n_classes)

	# Define opt functions
	def opt(self):
		learning_rate = 0.01
		self.cost = tf.losses.sparse_softmax_cross_entropy(labels=self.out, logits=self.logits)
		self.optimiser = tf.train.AdamOptimizer( learning_rate ).minimize(self.cost)


# Pretty shit atm, massive clusterfuck of hyperparameter usage (some inside model, some inside fitting here)
# Needs to be wrapped in larger class, and to understand how to feed in kwargs dict to feed_dict (i.e. with strings)
def fit_model( model, data, **kwargs ):
	train_inp, train_out, test_inp, test_out = data
	init_op = tf.global_variables_initializer()
	local_op = tf.local_variables_initializer()
	config = tf.ConfigProto( allow_soft_placement = True)


	# Below is a specific,ish, model fitting routine so we need to check that the model comes with appropriate hyperparameters to use it
	# We make local copies so we don't overwrite what is in the model already
	try:
		batch_size = model.batch_size
		epochs = model.epochs
	except:
		print("\nWARNING: Model " + model.name + " lacks appropriate hyper parameters, resorting test-case defaults\n")
		batch_size = int(round(0.05*len(train_inp)))
		epochs = 1000

	with tf.Session(config=config) as sess:
	    sess.run(init_op)
	    sess.run(local_op)

	    for epoch in range(epochs):
	            batch_pos = random.sample(range(0,len(train_inp)), batch_size)

	            with tf.name_scope("Batch_selection"):
	                batch_x = train_inp[batch_pos]
	                batch_y = train_out[batch_pos]
	            
	            _, c = sess.run([model.optimiser, model.cost], feed_dict={model.inp: batch_x, model.out: batch_y})

	            if(epoch%1 == 0):
	              batch_train_predict =  np.argmax(sess.run(model.logits, feed_dict={model.inp: batch_x }), axis = 1)
	              test_predict =  np.argmax(sess.run(model.logits, feed_dict={model.inp: test_inp}), axis = 1)
	            # 
	              batch_train_acc = fns.my_acc(batch_train_predict,batch_y)
	              test_acc = fns.my_acc(test_predict,test_out)

	              print(epoch,c, round(batch_train_acc, 2) , round(test_acc,2))
	 
