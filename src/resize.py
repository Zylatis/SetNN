import numpy as np
import os
from PIL import Image # gives better output control than matplotlib

def resize_img(im, target_size, save_path = None):

	old_size = im.size  # old_size[0] is in (width, height) format

	ratio = float(target_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])

	im = im.resize(new_size, Image.ANTIALIAS)

	new_im = Image.new("RGB", (target_size, target_size))
	new_im.paste(im, ((target_size-new_size[0])//2,
	                    (target_size-new_size[1])//2))	

	if save_path != None:
		new_im.save( save_path )

	return new_im

if __name__ == "__main__":
	imgs_folder = "../imgs/"
	count = 0
	for i in os.listdir( imgs_folder + "isolated/"):
		if i.endswith('.png'):

			im = Image.open( imgs_folder + "isolated/"+str(i) )
			resize_img(im, 128, imgs_folder + "processed/proc_img_ "+ str(i))
			count = count + 1

