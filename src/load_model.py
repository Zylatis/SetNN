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

imgs_folder = "../imgs/"
def yield_files():
	for i in os.listdir( imgs_folder + "processed/"):
		if i.endswith('.png'):
			# label =  ("_").join(  (i[:-4]).split('_')[2:6] ).strip()
			im = np.asarray(Image.open( imgs_folder + "processed/"+str(i) )).astype(np.uint8)
			yield [im, list(map(str.strip,(i[:-4]).split('_')[2:6])),i ]

hyperpars = {
'drop_rate':0.0,
'dense_size' : 128
}
cnn = models.CNN((128,128,3), 3, hyperpars, name = "test")
cnn.build_layers()
original_files = yield_files()
colour = ['red', 'purple', 'green']
count = ['single','triple','double']
shape = ['pill', 'diamond', 'squiggle']
fill = ['empty', 'grid', 'solid']


model_name = 'fill'
v = fill

saver = tf.train.Saver()
acc = 0
n = 0
with tf.Session() as sess:
	saver.restore(sess, tf.train.latest_checkpoint("../models/" +  model_name +"/"))
	for im in original_files:
		prediction = sess.run(cnn.logits, feed_dict={cnn.inp: [im[0]],  cnn.training: False})
		if v[np.argmax(prediction[0])] in im[1]:
			acc +=1
		print(v[np.argmax(prediction[0])], im[1])#,im[2])
		n+=1

print(acc/(1.*n))
# 	prediction = sess.run(cnn.logits, feed_dict={cnn.inp: x,  cnn.training: False})
# colours = ['red', 'purple', 'green']

# print(colours[np.argmax(prediction[0])], y)
