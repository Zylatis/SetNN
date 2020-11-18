import imgaug as ia
import numpy as np
import os
import random
import classes
import sys
import time 
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image # gives better output control than matplotlib
from pprint import pprint 
from imgaug import augmenters as iaa
from tqdm.contrib.concurrent import process_map 
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, ProcessPoolExecutor

# TODO put in config/env vars
RESIZED_LOC = "../data/train/raw_resized"
AUG_LOC = "../data/train/augmented"
RAW_LOC = "../data/train/raw"
FILE_EXTS = ['.png']

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

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


# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
	[
		# apply the following augmenters to most images
		iaa.Fliplr(0.5), # horizontally flip 50% of all images
		iaa.Flipud(0.2), # vertically flip 20% of all images
		
		sometimes(iaa.Affine(
			translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
			rotate=(-90, 90), # rotate by -45 to +45 degrees
			shear=(-10, 10) # shear by -16 to +16 degrees
		)),
		# execute 0 to 5 of the following (less important) augmenters per image
		# don't execute all of them, as that would often be way too strong
		iaa.SomeOf((0, 3),
			[
				# sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
				iaa.OneOf([
					iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
					iaa.AverageBlur(k=(2, 4)), # blur image using local means with kernel sizes between 2 and 7
					iaa.MedianBlur(k=(3, 7)), # blur image using local medians with kernel sizes between 2 and 7
				]),
				iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1)), # sharpen images
				iaa.Emboss(alpha=(0, 0.5), strength=(0, 0.1)), # emboss images
			
				iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.15), # add gaussian noise to images
				iaa.OneOf([
					iaa.Dropout((0.01, 0.1), per_channel=0.1), # randomly remove up to 10% of the pixels
					iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05)),
				]),
				iaa.Add((-10, 10), per_channel=0.1), # change brightness of images (by -10 to 10 of original value)
	
				#iaa.ContrastNormalization((0.75, 1.)), # improve or worsen the contrast
				sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
				sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
			],
			random_order=True
		)
	],
	random_order=True
)

def augment_img(inp):
	img_filename, n_replicates, class_vec_map = inp
	vec_label =  class_vec_map["_".join(img_filename[:-4].split("_")[:4])]
	img = np.asarray(Image.open(f"{RESIZED_LOC}/{img_filename}")).astype(np.uint8)

	replicated_data = np.asarray([ img for i in range(n_replicates)])
	images_aug = seq.augment_images( replicated_data )

	filenames = []
	for i in range(n_replicates):
		im = Image.fromarray(images_aug[i])
		im.save( f"../data/train/augmented/{img_filename}_{i}.png")
		filenames.append(f"{img_filename}_{i}.png")

	# TODO: refactor this giant turd
	return list(zip(filenames,[vec_label[0]]*n_replicates, [vec_label[1]]*n_replicates, [vec_label[2]]*n_replicates, [vec_label[3]]*n_replicates))

def resize_training_images(train_folder, rezised_train_folder, target_size):
	raw_files = os.listdir(train_folder)
	print("Resizing training images prior to augmentation:")
	for file in tqdm(raw_files):
		im = Image.open( f"{train_folder}/{file}")
		resize_img(im, target_size, f"{rezised_train_folder}/{file}")

	return len(raw_files)

if __name__ == '__main__':
	print("="*100)
	print("BEGIN")
	print("="*100)	
	
	# Map from free text to our class vectors (class_map is deprecate)
	class_map, class_vec_map = classes.get_labels()

	# Location where raw, isolated, labelled, training images are stored
	n_raw = resize_training_images(RAW_LOC, RESIZED_LOC, 128)
	n_proc = 12
	n_replicates = 50

	all_vec_labels = np.empty(shape=(1,4))
	total = n_raw*n_replicates

	print("Augmenting images and saving")

	file_list = os.listdir(RAW_LOC)
	aug_file_names = list(range(total))

	inputs = list(zip(file_list, [n_replicates]*n_raw, [class_vec_map]*n_raw))
	try:
		print("Flushing existing augmented files:")
		shutil.rmtree(AUG_LOC)
		os.mkdir(AUG_LOC)
	except:
		# Faster to try/except delete and create folders than delete individual files
		pass
	
	print(f"Beginning augmentation of {n_raw} images with {n_replicates} for total of {total}")
	results = []
	for result in process_map(augment_img, inputs, max_workers=n_proc, chunksize = 1):
		for x in result:
			results.append(list(x))

	print("Saving labels")
	results = pd.DataFrame(results)
	results.to_csv(f"{AUG_LOC}/aug_vec_labels.csv", index = False)

	print("="*100)
	print("COMPLETE")
	print("="*100)
