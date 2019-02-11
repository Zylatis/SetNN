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

imgs_folder = "../imgs/aug_imgs/"

class_map, inverse_class_map = classes.get_labels()


labels = np.loadtxt( "../imgs/labels.dat").astype(np.int32)
n = len(labels)
labels = labels[:n]

n_data = len(labels)
n_classes = len(class_map)

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

x_dim, y_dim, n_channels = im.shape

imgs = np.asarray(imgs)
labels = np.asarray(labels)
img_train, img_test, class_train, class_test = sk.train_test_split(imgs,labels,test_size=0.1 )
n_train = len(img_train)

learning_rate = 0.001
epochs = 2000

# batch_size = int(round(0.1*n_train))
batch_size = 12

print("### Setup ###")
print("Number of inputs: ", n_data)
print("Number of classes: ", n_classes)
print("X: ", x_dim, ", Y: ", y_dim)
print("Batch size: ", batch_size )
print(" Number epochs: ", epochs)
print("Test size:", len(class_test))
with tf.name_scope("Input"):
    inp = tf.placeholder(tf.float32, [None, x_dim,y_dim,n_channels], name = "input")

with tf.name_scope("Output"):
    out = tf.placeholder(tf.int32, [None], name = "output")


with tf.name_scope("layers"):
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=inp,
      filters=16,
      data_format = 'channels_last',
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      data_format = 'channels_last',
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat =tf.reshape(pool2, [-1, int(x_dim/4) * int(y_dim/4) * 32])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units = n_classes)


with tf.name_scope("Cost"):
    cost = tf.losses.sparse_softmax_cross_entropy(labels=out, logits=logits)
    acc, acc_op = tf.metrics.accuracy(labels=out, predictions=tf.argmax(logits, 1))

with tf.name_scope("Train"):
    optimiser = tf.train.AdamOptimizer( learning_rate ).minimize(cost)

# print inp
# print out
# print conv1
# print pool1
# print conv2
# print pool2
# print pool2_flat
# print dense
# print logits
# exit(0)
print("######################################")
init_op = tf.global_variables_initializer()
local_op = tf.local_variables_initializer()
# other_op = tf.contrib.layers.xavier_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto( allow_soft_placement = True)

with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_op)
    # sess.run(other_op)
    # summaryMerged = tf.summary.merge_all()
    # filename = "tensorboards/"
    # writer = tf.summary.FileWriter(filename, sess.graph) 

    for epoch in range(epochs):
            batch_pos = random.sample(range(0,n_train), batch_size)

            with tf.name_scope("Batch_selection"):
                batch_x = img_train[batch_pos]
                batch_y = class_train[batch_pos]
            

            _, c = sess.run([optimiser, cost], feed_dict={inp: batch_x, out: batch_y})
            if(epoch%100 == 0):
                train_acc =  sess.run(acc, feed_dict={inp: batch_x, out: batch_y})
                test_acc =  sess.run(acc, feed_dict={inp: img_test, out: class_test})
                print(epoch, c,round(train_acc,2),round(test_acc,2))

    train_predict =  sess.run(logits, feed_dict={inp: img_train, out: class_train})
    test_predict =  sess.run(logits, feed_dict={inp: img_test, out: class_test})


summary_train = pd.DataFrame( index = range(n_train), columns = ["train_actual", "train_predict"])
summary_train["train_actual"] = class_train
summary_train["train_predict"] = np.argmax(train_predict, axis = 1)
summary_train.to_csv("train_summary.csv")

summary_test = pd.DataFrame( index = range(n_data - n_train), columns = ["test_actual", "test_predict"])
summary_test["test_actual"] = class_test
summary_test["test_predict"] = np.argmax(test_predict, axis = 1)
summary_test.to_csv("test_summary.csv")

print(summary_train)
print(summary_test)
