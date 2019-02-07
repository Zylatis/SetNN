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


imgs_folder = "../imgs/aug_imgs/"
class_map, inverse_class_map = classes.get_labels()

labels = np.loadtxt( "../imgs/labels.dat").astype(np.int32)
n_data = len(labels)

n_classes = len(class_map)
imgs = []


for i in range(n_data):
    im = np.asarray(Image.open( imgs_folder +str(i) + '.png' ))
    im = tf.convert_to_tensor(im, dtype=tf.float32)
    imgs.append(im)

x_dim, y_dim, n_channels = im.shape
print im.shape
# manually check if labels line up 
# for label in labels:
	# print inverse_class_map[label]


# img_train, img_test, class_train, class_test = sk.train_test_split(imgs,labels,test_size=0.10 )
# scaler = StandardScaler().fit(img_train)
# input_layer = tf.reshape(img_train, [-1, x_dim, y_dim, n_channels])

learning_rate = 0.1
with tf.name_scope("Input"):
	inp = tf.placeholder(tf.float32, [None, x_dim,y_dim,n_channels], name = "input")

with tf.name_scope("Output"):
	out = tf.placeholder(tf.int32, [n_classes], name = "output")


with tf.name_scope("layers"):
	# # Convolutional Layer #1
	conv1 = tf.layers.conv2d(
	  inputs=inp,
	  filters=12,
	  data_format = 'channels_last',
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu)

	# # Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

	# # Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
	  inputs=pool1,
	  filters=24,
	  kernel_size=[5, 5],
	  padding="same",
	  data_format = 'channels_last',
	  activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 4], strides=4)
	pool2_flat = tf.reshape(pool2, [-1, 16*16*24])
	dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=0.4)
	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units = n_classes)


with tf.name_scope("Cost"):
	cost = tf.losses.sparse_softmax_cross_entropy(labels=out, logits=logits)
	
with tf.name_scope("Train"):
# add optimizer
	optimiser = tf.train.AdamOptimizer( learning_rate ).minimize(cost)

# # Dense Layer

print inp
print conv1
print pool1
print conv2
print pool2
print pool2_flat
print dense
# print logits
