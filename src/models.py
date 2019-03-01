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
		self.x_dim, self.y_dim, self.n_channels = im_shape
		self.n_classes = n_classes
		self.hyperpars = hyperpars
		print("Setup model: "),
	
		self.training = tf.placeholder(tf.bool, name='training')
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
		dense_size = self.dense_size  
		reg = 0.01
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
			activation=tf.nn.relu,
			kernel_regularizer=tf.contrib.layers.l1_regularizer( reg )
			)

		# Pooling Layer #1
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
		pool1_norm =tf.layers.batch_normalization(inputs = pool1, training = self.training)
		pool1_dropout = tf.layers.dropout( inputs=pool1_norm , rate=self.drop_rate )
		# Convolutional Layer #2 and Pooling Layer #2
		conv2 = tf.layers.conv2d(
			inputs=pool1_dropout ,
			filters=conv2_filters,
			kernel_size=[5, 5],
			padding="same",
			data_format = 'channels_last',
			activation=tf.nn.relu,
			kernel_regularizer=tf.contrib.layers.l1_regularizer(reg )
			)
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
		pool2_norm = tf.layers.batch_normalization(inputs = pool2, training = self.training)
		pool2_flat =tf.reshape(pool2_norm, [-1, int(self.x_dim/4) * int(self.y_dim/4) * conv2_filters])
		dense = tf.layers.dense(inputs=pool2_flat, units=dense_size, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l1_regularizer(reg )
		)
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
	model.opt()

	myVar_tf = tf.placeholder(dtype=tf.float32)
	tf.summary.scalar('cost', model.cost)
	tf.summary.scalar('train_acc',myVar_tf)
	merged = tf.summary.merge_all()
	train_inp, train_out, test_inp, test_out = data
	init_op = tf.global_variables_initializer()
	local_op = tf.local_variables_initializer()
	config = tf.ConfigProto( allow_soft_placement = True)
	
	print("Training "  + model.name)
	# Below is a specific,ish, model fitting routine so we need to check that the model comes with appropriate hyperparameters to use it
	# We make local copies so we don't overwrite what is in the model already
	try:
		batch_size = model.batch_size
		epochs = model.epochs
	except:
		print("\nWARNING: Model " + model.name + " lacks appropriate hyper parameters, resorting test-case defaults\n")
		batch_size = int(round(0.05*len(train_inp)))
		epochs = 10
	
	
	batches = fns.make_batches(train_inp, train_out, batch_size)
	saver = tf.train.Saver()
	with tf.Session(config=config) as sess:
		sess.run(init_op)
		sess.run(local_op)
		writer = tf.summary.FileWriter("../models/" + model.name, sess.graph)
		conv_count = 0
		step = 0
		for epoch in range(1, epochs):
			for batch in batches:
				_, c = sess.run([model.optimiser, model.cost], feed_dict={model.inp: batch[0], model.out: batch[1], model.training: True})

				dropout_save = model.drop_rate	
				model.drop_rate = 0. # for accuracy tests
				batch_train_predict =  np.argmax(sess.run(model.logits, feed_dict={model.inp: batch[0] ,model.training: True }), axis = 1)
				test_predict =  np.argmax(sess.run(model.logits, feed_dict={model.inp: test_inp, model.training: False}), axis = 1)
				  
				batch_train_acc = fns.my_acc(batch_train_predict, batch[1])
				test_acc = fns.my_acc(test_predict,test_out)
				
				summary = sess.run(merged , feed_dict={model.inp: batch[0], model.out: batch[1], model.training: True, myVar_tf : batch_train_acc})
				writer.add_summary(summary, step)
				model.drop_rate = dropout_save	
				
				if step % 10 == 0:
					print(epoch,step,c, round( batch_train_acc, 2),  round( test_acc, 2), conv_count )
				
				step += 1
			
				if(batch_train_acc >= 0.90):
					conv_count += 1
				else:
					conv_count = 0
				if conv_count >= 10:
					break
		save_path = saver.save(sess, "../models/" +  model.name +"/" + model.name + ".ckpt")
	tf.reset_default_graph()
	
