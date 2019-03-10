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
print("Loading " + str( n_data ) + " images: "),

imgs = []
for i in range(n_data):
	im = np.asarray(Image.open( imgs_folder +str(i) + '.png' ))#.convert('LA'))
	imgs.append(im)
	img_mean = int(np.mean(im.flatten()))
	im = im - img_mean
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
'learning_rate' : 0.01
}

epochs = 5
test = models.CNN_multi(im.shape, 3, hyperpars, name = "test")
test.build_branch(64,[16,32],'colour')
test.build_branch(128,[16,32],'counts')
test.build_branch(512,[16,32],'fill')
test.build_branch(512,[16,32],'shape')
test.opt()

init_op = tf.global_variables_initializer()
local_op = tf.local_variables_initializer()
config = tf.ConfigProto( allow_soft_placement = True)
batches = fns.make_batches(img_train, class_train, 50)
print len(class_train)
exit(0)
saver = tf.train.Saver()
with tf.Session(config = config) as sess:
	sess.run([init_op,local_op])
	for epoch in range(1, epochs):
		for batch in batches:

			batch_summary = {}			
			_,c = sess.run( [ test.optimiser, test.total_cost ], feed_dict={ test.inp: batch[0], test.out: batch[1], test.training: True} )
			print c

			
			for branch, output in test.logits.items():
				batch_train_predict = np.argmax(sess.run(output, feed_dict={test.inp: batch[0], test.training: True }), axis = 1)
				# print batch_train_predict
				# exit(0)
				batch_train_acc = fns.my_acc(batch_train_predict, batch[1][:,test.pos[branch]])
				print batch_train_predict
				print batch[1][:,test.pos[branch]]
				exit(0)

				# print(batch_train_acc)
				# exit(0)
				batch_summary[branch] = [batch_train_acc]
			print pd.DataFrame.from_dict(batch_summary)
			# batch_train_predict =  np.argmax(sess.run(test.logits['colour'], feed_dict={test.inp: batch[0], test.training: True }), axis = 1)
			# test_predict =  np.argmax(sess.run(test.logits, feed_dict={test.inp: test_inp, test.training: False}), axis = 1)

		
			  
		# 	batch_train_acc = fns.my_acc(batch_train_predict, batch[1])
		# 	test_acc = fns.my_acc(test_predict,test_out)
		
		# 	summary = sess.run(merged , feed_dict={ test.inp: batch[0], test.out: batch[1], test.training: True, myVar_tf : batch_train_acc })
		# 	writer.add_summary(summary, step)
		# 	log = np.concatenate((log,[[epoch, step, c, batch_train_acc, test_acc]]), axis = 0)

		# 	if step % 10 == 0:
		# 		print(epoch,step,c, round( batch_train_acc, 2),  round( test_acc, 2)  )
			
		# 	cost_history.append( c )
		# 	if fns.is_converged(cost_history, 100, 0.05) :
		# 		converged = True
		# 		break
		# 	step += 1
		# if converged:
		# 	break	
	save_path = saver.save(sess, "../models/multi_branch/multi_branch.ckpt")
tf.reset_default_graph()
# pos = 0
# cnn = models.CNN(im.shape, 3, hyperpars, name = "colour")
# cnn.build_layers()
# models.fit_model(cnn, [img_train,class_train[:,pos], img_test, class_test[:,pos]])
# del cnn

# exit(0)
# hyperpars['dense_size'] = 128


# pos = 1
# cnn = models.CNN(im.shape, 3, hyperpars, name = "count")
# cnn.build_layers()
# models.fit_model(cnn, [img_train,class_train[:,pos], img_test, class_test[:,pos]])
# del cnn


# hyperpars['dense_size']  = 512 #512

# pos = 2
# cnn = models.CNN(im.shape, 3, hyperpars, name = "fill")
# cnn.build_layers()
# cnn.opt()
# models.fit_model(cnn, [img_train,class_train[:,pos], img_test, class_test[:,pos]])


# pos = 3
# cnn = models.CNN(im.shape, 3, hyperpars, name = "shape")
# cnn.build_layers()
# cnn.opt()
# models.fit_model(cnn, [img_train,class_train[:,pos], img_test, class_test[:,pos]])
# del cnn



# else:
#   cnn = models.CNN(im.shape, 4, hyperpars, name = "CNN2")
#   cnn.build_layers()
#   cnn.opt()
#   models.fit_model(cnn, [img_train,class_train, img_test, class_test])
############## TESTS ##############
# def test_change():
#   # hyperpars don't matter here
#   hyperpars = {
#   'drop_rate':0.4,
#   'batch_size' : 64,
#   'learning_rate' : 0.1,
#   'epochs' : 100000
#   }

#   test_cnn = models.CNN(im.shape, n_classes, hyperpars)
#   test_cnn.build_layers()
#   test_cnn.opt()

#   config = tf.ConfigProto( allow_soft_placement = True)
#   init_op = tf.global_variables_initializer()
#   local_op = tf.local_variables_initializer()
#   batch_pos = random.sample(range(0,len(img_train)), hyperpars['batch_size'])
#   batch_x = img_train[batch_pos]
#   batch_y = class_train[batch_pos]
  
#   with tf.Session(config=config) as sess:
#     sess.run(init_op)
#     sess.run(local_op)

#     init_vals = sess.run(test_cnn.logits,feed_dict={test_cnn.inp: batch_x, test_cnn.out: batch_y})

#     sess.run([ test_cnn.optimiser], feed_dict={test_cnn.inp: batch_x, test_cnn.out: batch_y})
#     single_step_vals =  sess.run(test_cnn.logits,feed_dict={test_cnn.inp: batch_x, test_cnn.out: batch_y})

#     diff = single_step_vals - init_vals

#     assert (np.all(single_step_vals - single_step_vals==0))
#     assert (0 not in diff)
