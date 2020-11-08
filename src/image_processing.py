import imgaug as ia
import numpy as np
import os
import random
import classes
import sys
from imgaug import augmenters as iaa
from PIL import Image # gives better output control than matplotlib
# from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

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

def yield_files(class_vec_map):
	for i in os.listdir("../data/train/raw_resized"):
		if i.endswith('.png'):
			label =  "_".join(i[:-4].split("_")[:4])
			im = np.asarray(Image.open(f"../data/train/raw_resized/{i}")).astype(np.uint8)
			yield [im, class_vec_map[label]]

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
	
				iaa.ContrastNormalization((0.75, 1.)), # improve or worsen the contrast
				sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
				sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
			],
			random_order=True
		)
	],
	random_order=True
)

def augment_img(img, vec_label, n_replicates):

	replicated_data = np.asarray([ img for i in range(n_replicates)])
	images_aug = seq.augment_images( replicated_data )

	# all_vec_labels = np.concatenate( (all_vec_labels, [vec_label]*n_replicates))
	for i in range(n_replicates):
		im = Image.fromarray(images_aug[i])
		im.save( f"../data/train/augmented/{random.randint(0,100000000)}.png")
		# count += 1

	return [vec_label]*n_replicates

def resize_training_images(train_folder, rezised_train_folder, target_size):
	raw_files = os.listdir(train_folder)
	for file in tqdm(raw_files):

		im = Image.open( f"{train_folder}/{file}")
		resize_img(im, target_size, f"{rezised_train_folder}/{file}")

	return len(raw_files)

if __name__ == '__main__':
	# Map from free text to our class vectors (class_map is deprecate)
	class_map, class_vec_map = classes.get_labels()

	# Location where raw, isolated, labelled, training images are stored
	train_raw_folder = "../data/train/raw" 
	train_resized_folder = "../data/train/raw_resized"
	n_raw = resize_training_images(train_raw_folder, train_resized_folder, 128)
	
	n_replicates = 1
	count = 0
	img_generator = yield_files(class_vec_map)

	all_vec_labels = np.empty(shape=(1,4))
	total = n_raw*n_replicates

	print("Augmenting images and saving")

	executor = ThreadPoolExecutor(max_workers = 1)
	ckpt = []
	os.system("rm ../data/train/augmented/*")
	for img, vec_label in img_generator:
		ckpt.append(executor.submit(augment_img, img, vec_label, n_replicates))

	wait(ckpt, return_when = ALL_COMPLETED)
	r = [x.result() for x in ckpt]
	print(r)
	# 	replicated_data = np.asarray([ img for i in range(n_replicates)])
	# 	images_aug = seq.augment_images( replicated_data )

	# 	all_vec_labels = np.concatenate( (all_vec_labels, [vec_label]*n_replicates))

	# 	for i in range(n_replicates):
	# 		im = Image.fromarray(images_aug[i])
	# 		im.save( f"../data/train/augmented/{count}.png")
	# 		count += 1
		
	all_vec_labels =  np.delete(all_vec_labels,(0),axis = 0)
	print("Saving labels")
	np.savetxt("../imgs/aug_imgs/aug_vec_labels.dat", all_vec_labels, fmt = "%d")