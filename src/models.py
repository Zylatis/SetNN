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
import os

# Turn off TF reporting all sorts of CUDA reporting things
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class CNN:
	def __init__(self, im_shape, n_classes, hyperpars, **kwargs):
		self.x_dim = im_shape[0]
		self.y_dim = im_shape[1]

		if len(im_shape) == 2:
			self.inp_shape = [ None ] + list(im_shape) + [ 1 ]
		else:
			self.inp_shape = [ None ] + list(im_shape)

		self.n_classes = n_classes
		self.hyperpars = hyperpars
		print("Setup model: "),
	
		self.training = tf.placeholder(tf.bool)
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
		dense_size = self.dense_size  
		reg = 0.01
		# Placeholders for input/output (fed from feed_dict)	 
		self.inp = tf.placeholder(tf.float32, self.inp_shape, name = "input")
		self.out = tf.placeholder(tf.int32, [None], name = "output")

		# Convolutional Layer #1
		conv1 = tf.layers.conv2d(
			inputs=self.inp,
			filters=self.conv_filters[0],
			data_format = 'channels_last',
			kernel_size=[5,5],
			padding="same",
			activation=tf.nn.relu,
			kernel_regularizer=tf.contrib.layers.l1_regularizer( reg )
			)

		# Pooling Layer #1
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
		# pool1 = tf.layers.average_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
		pool1_norm =tf.layers.batch_normalization(inputs = pool1, training = self.training)
		pool1_dropout = tf.layers.dropout( inputs=pool1_norm, rate=self.drop_rate, training = self.training )

		# Convolutional Layer #2
		conv2 = tf.layers.conv2d(
			inputs=pool1_dropout,
			filters=self.conv_filters[1],
			data_format = 'channels_last',
			kernel_size=[5,5],
			padding="same",
			activation=tf.nn.relu,
			kernel_regularizer=tf.contrib.layers.l1_regularizer( reg )
			)

		# Pooling Layer #2
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
		# pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
		pool2_norm =tf.layers.batch_normalization(inputs = pool2, training = self.training)
		pool2_dropout = tf.layers.dropout( inputs=pool2_norm , rate=self.drop_rate, training = self.training )

		# # Convolutional Layer #3 
		# conv3 = tf.layers.conv2d(
		# 	inputs=pool2_dropout ,
		# 	filters=conv3_filters,
		# 	kernel_size=[5, 5],
		# 	padding="same",
		# 	data_format = 'channels_last',
		# 	activation=tf.nn.relu,
		# 	kernel_regularizer=tf.contrib.layers.l1_regularizer( reg )
		# 	)

		# pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
		# pool3_norm = tf.layers.batch_normalization(inputs = pool3, training = self.training)
		# pool3_dropout = tf.layers.dropout( inputs=pool3_norm , rate=self.drop_rate, training = self.training )
		
		# flat =tf.reshape(pool3_dropout, [-1, int(self.x_dim/8) * int(self.y_dim/8) * conv3_filters])
		flat =tf.reshape(pool2_dropout, [-1, int(self.x_dim/4) * int(self.y_dim/4) * self.conv_filters[1]])
		dense = tf.layers.dense(inputs=flat, units=dense_size, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l1_regularizer( reg ))
		dense_dropout = tf.layers.dropout( inputs=dense, rate=self.drop_rate )
		self.logits = tf.layers.dense(inputs=dense_dropout, units = self.n_classes,activation=tf.nn.relu)
		
	
	# Define opt functions
	def opt(self):
		self.cost = tf.losses.sparse_softmax_cross_entropy(labels=self.out, logits=self.logits) + tf.losses.get_regularization_loss()
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.optimiser = tf.train.AdamOptimizer( self.learning_rate ).minimize( self.cost )


# Pretty shit atm, massive clusterfuck of hyperparameter usage (some inside model, some inside fitting here)
# Needs to be wrapped in larger class, and to understand how to feed in kwargs dict to feed_dict (i.e. with strings)
def fit_model( model, data, **kwargs ):
	log = np.array([[0,0,0,0,0]])
	model.opt()

	myVar_tf = tf.placeholder(dtype=tf.float32)
	tf.summary.scalar('cost', model.cost)
	tf.summary.scalar('train_acc',myVar_tf)

	merged = tf.summary.merge_all()
	train_inp, train_out, test_inp, test_out = data
	init_op = tf.global_variables_initializer()
	local_op = tf.local_variables_initializer()
	config = tf.ConfigProto( allow_soft_placement = True)
	cost_history = []
	print("Training "  + model.name)
	print("Hyperpars:")
	print(model.hyperpars)
	# Below is a specific,ish, model fitting routine so we need to check that the model comes with appropriate hyperparameters to use it
	# We make local copies so we don't overwrite what is in the model already
	try:
		batch_size = model.batch_size
		epochs = model.epochs
	except:
		print("\nWARNING: Model " + model.name + " lacks appropriate hyper parameters, resorting test-case defaults\n")
		batch_size = int(round(0.05*len(train_inp)))
		epochs = 10
	
	converged = False
	batches = fns.make_batches(train_inp, train_out, batch_size)
	saver = tf.train.Saver()#tf.trainable_variables() + model.pool1_norm.moving_variance + model.pool1_norm.moving_mean +model.pool1_norm.moving_mean + model.pool2_norm.moving_variance
	with tf.Session(config = config) as sess:
		sess.run(init_op)
		sess.run(local_op)
		writer = tf.summary.FileWriter("../models/" + model.name, sess.graph)
		conv_count = 0
		step = 0
		for epoch in range(1, epochs):
			for batch in batches:
				_, c = sess.run( [ model.optimiser, model.cost ], feed_dict={ model.inp: batch[0], model.out: batch[1], model.training: True} )
				
				batch_train_predict =  np.argmax(sess.run(model.logits, feed_dict={model.inp: batch[0], model.training: True }), axis = 1)
				test_predict =  np.argmax(sess.run(model.logits, feed_dict={model.inp: test_inp, model.training: False}), axis = 1)
				  
				batch_train_acc = fns.my_acc(batch_train_predict, batch[1])
				test_acc = fns.my_acc(test_predict,test_out)
			
				summary = sess.run(merged , feed_dict={ model.inp: batch[0], model.out: batch[1], model.training: True, myVar_tf : batch_train_acc })
				writer.add_summary(summary, step)
				log = np.concatenate((log,[[epoch, step, c, batch_train_acc, test_acc]]), axis = 0)

				if step % 10 == 0:
					print(epoch,step,c, round( batch_train_acc, 2),  round( test_acc, 2)  )
				
				cost_history.append( c )
				if fns.is_converged(cost_history, 100, 0.05) :
					converged = True
					break
				step += 1
			if converged:
				break	
		np.savetxt( "../models/" +  model.name +"/" + model.name + ".log", log, delimiter = "\t")
		save_path = saver.save(sess, "../models/" +  model.name +"/" + model.name + ".ckpt",)
	tf.reset_default_graph()
	


class CNN_multi:
	def __init__(self, im_shape, n_classes, hyperpars, **kwargs):
		self.x_dim = im_shape[0]
		self.y_dim = im_shape[1]

		if len(im_shape) == 2:
			self.inp_shape = [ None ] + list(im_shape) + [ 1 ]
		else:
			self.inp_shape = [ None ] + list(im_shape)

		self.n_classes = n_classes
		self.hyperpars = hyperpars
		print("Setup model: "),
	
		self.training = tf.placeholder(tf.bool)
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


	def build_conv_section( self, input, filters, kernel, activation, reg):
		conv = tf.layers.conv2d(
			inputs=input,
			filters=filters,
			data_format = 'channels_last',
			kernel_size= kernel ,
			padding="same",
			activation=activation,
			kernel_regularizer=tf.contrib.layers.l1_regularizer( reg )
			)

		pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
		pool_norm =tf.layers.batch_normalization(inputs = pool, training = self.training)
		pool_dropout = tf.layers.dropout( inputs=pool_norm, rate=self.drop_rate, training = self.training )

		return(pool_dropout)


	def build_branch(self, dense_size, conv_filters ):
		reg = 0.01
		# Placeholders for input/output (fed from feed_dict)	 
		inp = tf.placeholder(tf.float32, self.inp_shape, name = "input")
		out = tf.placeholder(tf.int32, [None], name = "output")

		layer1 = self.build_conv_section( inp, conv_filters[0], [5,5], tf.nn.relu, reg)
		layer2 = self.build_conv_section( layer1, conv_filters[1], [5,5], tf.nn.relu, reg)

		flat =tf.reshape(layer2, [-1, int(self.x_dim/4) * int(self.y_dim/4) * self.conv_filters[1]])
		dense = tf.layers.dense(inputs=flat, units=dense_size, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l1_regularizer( reg ))
		dense_dropout = tf.layers.dropout( inputs=dense, rate=self.drop_rate )
		logits = tf.layers.dense(inputs=dense_dropout, units = self.n_classes,activation=tf.nn.relu)

		# return(inp, out, logits)
	
	# Define opt functions
	# def opt(self):
	# 	self.cost = tf.losses.sparse_softmax_cross_entropy(labels=self.out, logits=self.logits) + tf.losses.get_regularization_loss()
	# 	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	# 	with tf.control_dependencies(update_ops):
	# 		self.optimiser = tf.train.AdamOptimizer( self.learning_rate ).minimize( self.cost )
