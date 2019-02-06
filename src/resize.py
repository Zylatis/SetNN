import numpy as np
import os
from scipy.misc import imresize
from skimage.measure import block_reduce
imgs_folder = "../imgs/"
from PIL import Image # gives better output control than matplotlib


# Shamelessly pinched but I get it anyway
def crop_center(img,cropx,cropy):
	y,x,c = img.shape
	startx = x/2 - cropx/2
	starty = y/2 - cropy/2    
	return img[ starty:starty + cropy, startx:startx + cropx, :]


imgs = []
target_x = 100
target_y = 100
count = 0
labels = []
for i in os.listdir( imgs_folder + "isolated/"):
	if i.endswith('.png'):

		im = np.asarray(Image.open( imgs_folder + "isolated/"+str(i) ))
		imgs.append(im)
		x,y,z = im.shape
		aspect = y/(1.*x)

		if x > y:
			new_size = (100,int(100*aspect))
			pad_y = int((100-new_size[1])/2.)
			pad_x = 0
		else:
			new_size =  (int(100/aspect),100)
			pad_x = int((100-new_size[0])/2.)
			pad_y = 0

		new_im = imresize(im, size = new_size)
		new_im = np.pad(new_im, pad_width = ((pad_x,pad_x),(pad_y,pad_y),(0,0)), mode = 'constant', constant_values = (255,255)) #
		
		im = Image.fromarray(new_im)
		im.save(imgs_folder + "processed/proc_img_ "+ str(i) )
		count = count + 1
