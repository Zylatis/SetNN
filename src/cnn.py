import numpy as np 
import os
from PIL import Image # gives better output control than matplotlib
import classes
imgs_folder = "../imgs/aug_imgs/"
class_map, inverse_class_map = classes.get_labels()

labels = np.loadtxt( "../imgs/labels.dat")
n_data = len(labels)

imgs = []
for i in range(n_data):
    im = np.asarray(Image.open( imgs_folder +str(i) + '.png' )).astype(np.uint8)
    imgs.append(im)

# manually check if labels line up 
# for label in labels:
	# print inverse_class_map[label]