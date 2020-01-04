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

imgs_folder = "../imgs/"
def yield_files():
	for i in os.listdir( imgs_folder + "processed/"):
		if i.endswith('.png'):
			# label =  ("_").join(  (i[:-4]).split('_')[2:6] ).strip()
			im = np.asarray(Image.open( imgs_folder + "processed/"+str(i) )).astype(np.uint8)
			yield [im, list(map(str.strip,(i[:-4]).split('_')[2:6])),i ]

original_files = yield_files()
colour = ['red', 'purple', 'green']
count = ['single','double', 'triple']
shape = ['pill', 'diamond', 'squiggle']
fill = ['empty', 'grid', 'solid']

model_name = 'colour'
v = colour

saver = tf.train.import_meta_graph("../models/" +  model_name +"/"+model_name+".ckpt.meta")
acc = 0
n = 0

with tf.Session() as sess:
	saver.restore(sess,tf.train.latest_checkpoint("../models/" +  model_name +"/"))
	graph = tf.get_default_graph()
	inp = graph.get_tensor_by_name("input:0")
	logits = graph.get_tensor_by_name("logits/BiasAdd:0")
	training = graph.get_tensor_by_name("training:0")
	
	for im in original_files:
		prediction = sess.run(logits, feed_dict={inp: [im[0]], training : False})

		if v[np.argmax(prediction[0])] in im[1]:
			acc +=1
		print(v[np.argmax(prediction[0])], im[1])#,im[2])
		
		
		labels = " ".join( v ) 
		scores = " ".join([ str(round(x/np.sum(prediction[0] ),2)) for x in prediction[0] ])

		im_show = Image.fromarray(im[0])
		draw = ImageDraw.Draw(im_show)
		draw.text((0, 0),  labels ,(0,0,0))
		draw.text((0, 10), scores ,(0,0,0))

		im_show.save( imgs_folder + "/check_original/" + str(n) + ".png")
		
		n+=1

print( acc/(1.*n))