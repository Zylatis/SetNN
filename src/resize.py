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
