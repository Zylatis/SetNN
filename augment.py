from imgaug import augmenters as iaa
import numpy as np
# import cv2
import os
import matplotlib
matplotlib.use('Agg')   
from matplotlib.image import imread
from matplotlib import pyplot as plt

imgs = []
for i in os.listdir("imgs/processed/"):
    if i.endswith('.png'):
        im = imread("imgs/processed/"+str(i))
        imgs.append(im)

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

images_aug = seq.augment_images(imgs)  # done by the library
for i in range(len(images_aug)):
    plt.imshow(images_aug[i], interpolation='nearest')
    plt.savefig("imgs/aug_imgs/aug_img" + str(i) +".png" )