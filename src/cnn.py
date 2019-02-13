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
def my_acc( v1, v2 ):
  count = 0.
  n = len(v1)
  for i in range(n):
    if v1[i] == v2[i]:
      count = count + 1.
  return count/(1.*n)

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

print("### Setup ###")
print("Number of inputs: ", n_data)
print("Number of classes: ", n_classes)
print("X: ", x_dim, ", Y: ", y_dim)
print("Batch size: ", batch_size )
print(" Number epochs: ", epochs)
print("Test size:", len(class_test))

cnn = model.Model(im.shape, n_classes)
cnn.basic_cnn()
cnn.opt()
init_op = tf.global_variables_initializer()
local_op = tf.local_variables_initializer()
# saver = tf.train.Saver()

config = tf.ConfigProto( allow_soft_placement = True)

with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_op)

    batch_pos = random.sample(range(0,n_train), batch_size)
    batch_x = img_train[batch_pos]
    batch_y = class_train[batch_pos]
    print sess.run(cnn.logits,feed_dict={cnn.inp: batch_x, cnn.out: batch_y})

    c = sess.run([ cnn.optimiser], feed_dict={cnn.inp: batch_x, cnn.out: batch_y})
    print sess.run(cnn.logits,feed_dict={cnn.inp: batch_x, cnn.out: batch_y})

exit(0)
# model.basic_cnn(x_dim, y_dim, n_channels, n_classes)

# with tf.name_scope("Cost"):
#     cost = tf.losses.sparse_softmax_cross_entropy(labels=out, logits=logits)

# with tf.name_scope("Train"):
#     optimiser = tf.train.AdamOptimizer( learning_rate ).minimize(cost)
print("######################################")
init_op = tf.global_variables_initializer()
local_op = tf.local_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto( allow_soft_placement = True)

with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_op)
 
    for epoch in range(epochs):
            batch_pos = random.sample(range(0,n_train), batch_size)

            with tf.name_scope("Batch_selection"):
                batch_x = img_train[batch_pos]
                batch_y = class_train[batch_pos]
            

            _, c = sess.run([optimiser, cost], feed_dict={inp: batch_x, out: batch_y, drop_rate: 0.4})

            

            if(epoch%100 == 0):
              batch_train_predict =  np.argmax(sess.run(logits, feed_dict={inp: batch_x, drop_rate: 0 }), axis = 1)
              test_predict =  np.argmax(sess.run(logits, feed_dict={inp: img_test, drop_rate: 0}), axis = 1)
            
              batch_train_acc = my_acc(batch_train_predict,batch_y)
              test_acc = my_acc(test_predict,class_test)

              print(epoch,c, round(batch_train_acc, 2) , round(test_acc,2))
             # if(batch_train_acc > 0.999):
                #print("Train acc > 0.999")
                # break
      
    # train_predict =  sess.run(logits, feed_dict={inp: img_train, out: class_train})
    # test_predict =  sess.run(logits, feed_dict={inp: img_test, out: class_test})



# summary_train = pd.DataFrame( index = range(n_train), columns = ["train_actual", "train_predict"])
# summary_train["train_actual"] = class_train
# summary_train["train_predict"] = np.argmax(train_predict, axis = 1)
# summary_train.to_csv("train_summary.csv",encoding="utf-8")

# summary_test = pd.DataFrame( index = range(n_data - n_train), columns = ["test_actual", "test_predict"])
# summary_test["test_actual"] = class_test
# summary_test["test_predict"] = np.argmax(test_predict, axis = 1)
# summary_test.to_csv("test_summary.csv",encoding="utf-8")

# print(summary_train)
# print(summary_test)
