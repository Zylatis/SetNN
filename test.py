import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt
import os
from scipy.misc import imresize
# from skimage.transform import resize
from skimage.measure import block_reduce

# Shamelessly pinched but I get it anyway
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x/2 - cropx/2
    starty = y/2 - cropy/2    
    return img[ starty:starty + cropy, startx:startx + cropx, :]


imgs = []
target_x = 100
target_y = 100
# d = crop_center(d, 170,100)
for i in os.listdir("imgs/isolated/"):
    if i.endswith('.png'):
    	# print i
    	im = imread("imgs/isolated/"+str(i))
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
    	print x,y
    	print pad_x,pad_y
    	new_im = imresize(im, size = new_size)
    	new_im = np.pad(new_im, pad_width = ((pad_x,pad_x),(pad_y,pad_y),(0,0)), mode = 'constant', constant_values = (0,0))
    	# print new_size,pad_x
    break


plt.imshow(new_im, interpolation='nearest')
plt.show()