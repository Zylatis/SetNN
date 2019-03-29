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
import models
import fns

# model = "vec_"
model= ""
print("### Setup ###")

imgs_folder = "../imgs/aug_imgs/"
# labels = np.loadtxt( "../imgs/aug_imgs/aug_"+model+"labels.dat").astype(np.int32)
vec_labels = np.asarray(np.loadtxt( "../imgs/aug_imgs/aug_vec_labels.dat").astype(np.int32))

# colour, count, shape, fill
n_data = len(vec_labels)
n_data = 1000
print("Loading " + str( n_data ) + " images: "),

imgs = []
for i in range(n_data):
	im = np.asarray(Image.open( imgs_folder +str(i) + '.png' ))#.convert('LA'))
	imgs.append(im)
	img_mean = int(np.mean(im.flatten()))
	# im = im - img_mean
	# im_show = Image.fromarray(im)
	# im_show.show()
	# exit(0)
	
print(im.shape)
print("Done.")
# random.seed(0)
imgs = np.asarray(imgs)
n = len( imgs )
img_train, img_test, class_train, class_test = sk.train_test_split( imgs[:n], vec_labels[:n], test_size = 0.1, shuffle = True )

img_train = img_train
img_test = img_test
class_train = class_train
class_test = class_test

hyperpars = {
'drop_rate' : 0.4,
'learning_rate' : 0.0001
}

epochs = 2
test = models.CNN_multi(im.shape, 3, hyperpars, name = "test")
test.build_branch(12,[16,32],'colour')
test.build_branch(12,[16,32],'counts')
test.build_branch(12,[16,32],'fill')
test.build_branch(12,[16,32],'shape')
test.opt()

init_op = tf.global_variables_initializer()
local_op = tf.local_variables_initializer()
config = tf.ConfigProto( allow_soft_placement = True)
batches = fns.make_batches(img_train, class_train, 256)
del img_train, class_train
step = 0
saver = tf.train.Saver()
merged = tf.summary.merge_all()
with tf.Session(config = config) as sess:
	sess.run(init_op)
	sess.run(local_op)
	writer = tf.summary.FileWriter("../models/multi_branch/", sess.graph)
	for epoch in range(1, epochs):
		for batch in batches:

			batch_summary = {}			
			_, c = sess.run( [ test.optimiser, test.total_cost  ], feed_dict={ test.inp: batch[0], test.out: batch[1], test.training: True} )
			

			# graph = tf.get_default_graph()
			# inp = graph.get_tensor_by_name("input:0")

			# colour_label = graph.get_tensor_by_name("colour/BiasAdd:0")
			# test = sess.run(  [ colour_label, test.logits['colour']], feed_dict={ test.inp: batch[0][:3],  test.training: True} )
			# print(test)
			# exit(0)
			if step % 10 == 0:
				summary = sess.run(merged , feed_dict={ test.inp: batch[0], test.out: batch[1], test.training: True })
				writer.add_summary(summary, step)
				for branch, output in test.logits.items():
					batch_train_predict = np.argmax(sess.run(output, feed_dict={test.inp: batch[0], test.training: True }), axis = 1)
					batch_train_acc = round(fns.my_acc(batch_train_predict, batch[1][:,test.pos[branch]]),2)

					test_predict = np.argmax(sess.run(output, feed_dict={test.inp: img_test, test.training: False }), axis = 1)
					test_acc = round(fns.my_acc(test_predict, class_test[:,test.pos[branch]]),2)

					batch_summary[branch] = [batch_train_acc, test_acc]
				print("Step ",step, "cost ", c)
				print(pd.DataFrame.from_dict(batch_summary))
				print("\n")
			step += 1
		
	save_path = saver.save(sess, "../models/multi_branch/multi_branch.ckpt")
tf.reset_default_graph()
