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
saver = tf.train.import_meta_graph("../models/multi_branch/multi_branch.ckpt.meta")
acc = 0
n = 0

labels = {
'colour' :['red', 'purple', 'green'],
'counts' : ['single','double', 'triple'],
'fill' : ['empty', 'grid', 'solid'],
'shape' :  ['pill', 'diamond', 'squiggle']
}

labels_flat = [ labels['colour'], labels['counts'], labels['fill'], labels['shape']]

with tf.Session() as sess:
	saver.restore(sess,tf.train.latest_checkpoint("../models/multi_branch/"))
	graph = tf.get_default_graph()
	inp = graph.get_tensor_by_name("input:0")

	colour_label = graph.get_tensor_by_name("logits_colour/BiasAdd:0")
	shape_label = graph.get_tensor_by_name("logits_shape/BiasAdd:0")
	fill_label = graph.get_tensor_by_name("logits_fill/BiasAdd:0")
	count_label = graph.get_tensor_by_name("logits_counts/BiasAdd:0")

	training = graph.get_tensor_by_name("training:0")

	for im in original_files:
		predicted_label_array = sess.run( [ colour_label, count_label, fill_label, shape_label ], feed_dict={inp: [im[0]], training : False})

		
		# if v[np.argmax(prediction[0])] in im[1]:
		# 	acc +=1
		# print(v[np.argmax(prediction[0])], im[1])#,im[2])
		
		# prediction = predicted_label_array[0]
		# labels = " ".join( v ) 
		# scores = " ".join([ str(round(x/np.sum(prediction[0] ),2)) for x in prediction[0] ])
	
		
		for arr in predicted_label_array:
			label_list = np.abs(arr[0]*(arr[0]>0))
			total = np.sum(label_list )

			print( [ str(round(x/total,2)) for x in label_list ])
		# print( [ str(round(x/np.sum(predicted_label_array[0] ),2)) for x in predicted_label_array[0] ] )
		exit(0)
		# im_show = Image.fromarray(im[0])
		# draw = ImageDraw.Draw(im_show)
		# draw.text((0, 0),  labels ,(0,0,0))
		# draw.text((0, 10), scores ,(0,0,0))

		# im_show.save( imgs_folder + "/check_original/" + str(n) + ".png")
		
		# n+=1

# print( acc/(1.*n))