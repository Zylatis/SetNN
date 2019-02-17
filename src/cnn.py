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
import model
import fns

print("### Setup ###")

imgs_folder = "../imgs/aug_imgs/"

labels = np.loadtxt( "../imgs/aug_imgs/aug_labels.dat").astype(np.int32)
n = len(labels)
# n = 60
print("Loading " + str(n) + " images:")
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
x_dim, y_dim, n_channels = im.shape
imgs = np.asarray(imgs)
labels = np.asarray(labels)
img_train, img_test, class_train, class_test = sk.train_test_split(imgs,labels,test_size=0.1 )
n_train = len(img_train)

learning_rate = 0.00001
epochs = 100000
batch_size = 64

print("Number of inputs: ", n_data)
print("Number of classes: ", n_classes)
print("X: ", x_dim, ", Y: ", y_dim)
print("Batch size: ", batch_size )
print(" Number epochs: ", epochs)
print("Test size:", len(class_test))

cnn = model.CNN(im.shape, n_classes)
cnn.build_layers()
cnn.opt()
init_op = tf.global_variables_initializer()
local_op = tf.local_variables_initializer()
config = tf.ConfigProto( allow_soft_placement = True)

with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_op)

    for epoch in range(epochs):
            batch_pos = random.sample(range(0,n_train), batch_size)

            with tf.name_scope("Batch_selection"):
                batch_x = img_train[batch_pos]
                batch_y = class_train[batch_pos]
            
            _, c = sess.run([cnn.optimiser, cnn.cost], feed_dict={cnn.inp: batch_x, cnn.out: batch_y, cnn.drop_rate: 0.4})

            

            if(epoch%100 == 0):
              batch_train_predict =  np.argmax(sess.run(cnn.logits, feed_dict={cnn.inp: batch_x, cnn.drop_rate: 0 }), axis = 1)
              test_predict =  np.argmax(sess.run(cnn.logits, feed_dict={cnn.inp: img_test, cnn.drop_rate: 0}), axis = 1)
            
              batch_train_acc = fns.my_acc(batch_train_predict,batch_y)
              test_acc = fns.my_acc(test_predict,class_test)

              print(epoch,c, round(batch_train_acc, 2) , round(test_acc,2))
 

def test_change():
  test_cnn = model.Model(im.shape, n_classes)
  test_cnn.basic_cnn()
  test_cnn.opt()

  config = tf.ConfigProto( allow_soft_placement = True)
  init_op = tf.global_variables_initializer()
  local_op = tf.local_variables_initializer()
  batch_pos = random.sample(range(0,n_train), batch_size)
  batch_x = img_train[batch_pos]
  batch_y = class_train[batch_pos]
  
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_op)

    init_vals = sess.run(cnn.logits,feed_dict={cnn.inp: batch_x, cnn.out: batch_y})

    sess.run([ cnn.optimiser], feed_dict={cnn.inp: batch_x, cnn.out: batch_y})
    single_step_vals =  sess.run(cnn.logits,feed_dict={cnn.inp: batch_x, cnn.out: batch_y})

    assert (0 in single_step_vals - single_step_vals)
    assert (0 not in single_step_vals - init_vals)
