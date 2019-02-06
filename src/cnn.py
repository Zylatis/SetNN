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

labels = np.loadtxt( "../imgs/labels.dat")
n_data = len(labels)

imgs = []
for i in range(n_data):
    im = np.asarray(Image.open( imgs_folder +str(i) + '.png' )).astype(np.int32)
    imgs.append(im)

x_dim, y_dim, n_channels = im.shape
# manually check if labels line up 
# for label in labels:
	# print inverse_class_map[label]
print type(imgs)


img_train, img_test, class_train, class_test = sk.train_test_split(imgs,labels,test_size=0.10 )
# scaler = StandardScaler().fit(img_train)
img_train = np.asarray(img_train)
input_layer = tf.reshape(img_train, [-1, x_dim, y_dim, 3])
#
