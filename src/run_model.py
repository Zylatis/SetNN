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

labels = np.loadtxt( "../imgs/aug_imgs/aug_labels.dat").astype(np.int32)
n = len(labels)
#n =1500
print("Loading " + str(n) + " images: "),
labels = labels[:n]
n_data = len(labels)
n_classes = max(labels)+1# very hacky 

empty = np.zeros(n_classes)
label_vectors = []
for label in labels:
    temp  = copy.deepcopy( empty )
    temp[label] = 1.
    label_vectors.append(temp)
label_vectors = np.array(label_vectors)

imgs = []
for i in range(n_data):
    im = np.asarray(Image.open( imgs_folder +str(i) + '.png' ))
    imgs.append(im)
    img_mean = int(np.mean(im.flatten()))
    im = im - img_mean

print("Done.")
print("Number of classes: " + str(n_classes) )
x_dim, y_dim, n_channels = im.shape
imgs = np.asarray(imgs)
labels = np.asarray(labels)
img_train, img_test, class_train, class_test = sk.train_test_split(imgs,labels,test_size=0.1 )

hyperpars = {
'drop_rate':0.4,
'batch_size' : 32,
'learning_rate' : 0.00001,
'epochs' : 100000
}

cnn = models.CNN(im.shape, n_classes, hyperpars, name = "CNN1")
cnn.build_layers()
cnn.opt()
models.fit_model(cnn, [img_train,class_train, img_test, class_test])

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
