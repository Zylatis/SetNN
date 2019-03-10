import numpy as np 
import os
from PIL import Image, ImageFont, ImageDraw # gives better output control than matplotlib
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
import cv2



cap = cv2.VideoCapture(0)

hyperpars = {
'drop_rate':0.0,
'dense_size' : 64,
'conv_filters' : [ 16, 32 ]
}


cnn = models.CNN((128,128,3), 3, hyperpars, name = "test")

# test = models.CNN_multi((128,128,3), 3, hyperpars, name = "test")
# test.build_branch(128,[16,32])
# exit(0)
cnn.build_layers()
# cnn.training = False
# original_files = yield_files()

colour = ['red', 'purple', 'green']
count = ['single','double', 'triple']
shape = ['pill', 'diamond', 'squiggle']
fill = ['empty', 'grid', 'solid']


model_name = 'colour'
v = colour

saver = tf.train.Saver()
acc = 0
n = 0
# font = ImageFont.truetype("arial.ttf", 8)
with tf.Session() as sess:
	saver.restore(sess, tf.train.latest_checkpoint("../models/" +  model_name +"/"))

	while 1:
		ret, im = cap.read()
		im = cv2.resize(im , (128,128), interpolation = cv2.INTER_AREA)
		prediction = sess.run(cnn.logits, feed_dict={cnn.inp: [im], cnn.training : False})
		
		print(v[np.argmax(prediction[0])])
		
		exit(0)
		k = cv2.waitKey(30)
		if k == 27:
			break
