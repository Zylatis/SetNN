import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt


# Shamelessly pinched but I get it anyway
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x/2 - cropx/2
    starty = y/2 - cropy/2    
    return img[ starty:starty + cropy, startx:startx + cropx, :]


imgs = []
n_images = 4
for i in range(n_images):
	im = imread('imgs/'+str(i)+'.png');
	im = crop_center(im, 170,100)
	imgs.append(im)
	print im.shape

# a = crop_center(a, 170,100)
# b = crop_center(b, 170,100)
# c = crop_center(c, 170,100)
# d = crop_center(d, 170,100)
# plt.imshow(b, interpolation='nearest')
# plt.show()	