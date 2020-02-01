import numpy as np
import os
from PIL import Image # gives better output control than matplotlib

imgs_folder = "../imgs/"
imgs = []
target_x, target_y = 128, 128
count = 0
for i in os.listdir( imgs_folder + "isolated/"):
	if i.endswith('.png'):

		im = Image.open( imgs_folder + "isolated/"+str(i) )

		desired_size = 128
		old_size = im.size  # old_size[0] is in (width, height) format

		ratio = float(desired_size)/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])
	
		im = im.resize(new_size, Image.ANTIALIAS)
	
		new_im = Image.new("RGB", (desired_size, desired_size))
		new_im.paste(im, ((desired_size-new_size[0])//2,
		                    (desired_size-new_size[1])//2))
		
		new_im.save(imgs_folder + "processed/proc_img_ "+ str(i) )
		count = count + 1