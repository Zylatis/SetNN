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

print("### Setup ###")

imgs_folder = "../imgs/aug_imgs/"
vec_labels = np.asarray(np.loadtxt( "../imgs/aug_imgs/aug_vec_labels.dat").astype(np.int32))

# colour, count, fill, shape
n_data = len(vec_labels)
# n_data = 1000
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

imgs = np.asarray(imgs)
n = len( imgs )
img_train, img_test, class_train, class_test = sk.train_test_split( imgs[:n], vec_labels[:n], test_size = 0.1, shuffle = True )

img_train = img_train
img_test = img_test
class_train = class_train
class_test = class_test

hyperpars = {
'drop_rate' : 0.4,
'learning_rate' : 0.0001,
'dense_size' : 64,
"conv_filters" : [16,32],
'batch_size' : 512,
"epochs" : 25
}

pos = 0
cnn = models.CNN(im.shape, 3, hyperpars, name = "colour")
cnn.build_layers()
models.fit_model(cnn, [img_train,class_train[:,pos], img_test, class_test[:,pos]])
del cnn

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
