import imgaug as ia
import numpy as np
import os
import random
import classes
import sys
from imgaug import augmenters as iaa
from PIL import Image # gives better output control than matplotlib

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

imgs_folder = "../imgs/"
class_map, class_vec_map = classes.get_labels()

def yield_files():
	for i in os.listdir( imgs_folder + "processed/"):
		if i.endswith('.png'):
			label =  ("_").join(  (i[:-4]).split('_')[2:6] ).strip()
			class_val = class_map[label]
			class_vec_val = class_vec_map[label]
			# classes_seen.append(class_val)
			im = np.asarray(Image.open( imgs_folder + "processed/"+str(i) )).astype(np.uint8)
			yield [im, class_val, class_vec_val]

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

n_raw = 0
for i in os.listdir( imgs_folder + "processed/"):
	n_raw +=1
if n_raw == 0:
	print("Run resize.py first, dumbarse.")
	exit(0)

n_replicates = 150
count = 0
img_generator = yield_files()
all_labels = np.asarray([])
all_vec_labels = np.empty(shape=(1,4))
total = n_raw*n_replicates



seenc = []
sc = 0
print("Augmenting images and saving")
for img, label, vec_label in img_generator:

	if label in seenc:
		sc += 1
	else:
		seenc.append(label)

	sys.stdout.flush() 
	replicated_data = np.asarray([ img for i in range(n_replicates)])
	images_aug = seq.augment_images( replicated_data )
	all_labels = np.concatenate( (all_labels, [label]*n_replicates))

	all_vec_labels = np.concatenate( (all_vec_labels, [vec_label]*n_replicates))

	for i in range(n_replicates):
		im = Image.fromarray(images_aug[i])
		im.save( imgs_folder + "aug_imgs/" + str( count) +".png")
		count += 1
	perc = int(round(100.*count/(1.*total)))
	print("\rProgress: " +str(perc) + "%"),

all_vec_labels =  np.delete(all_vec_labels,(0),axis = 0)

classes_seen = sorted(all_labels)
classes_seen = list(set(classes_seen))
n_seen = len(classes_seen)
restricted_map = {}

for i in range(n_seen):
	restricted_map[int(classes_seen[i])] = i

print("\nActual number of classes represented in data: " + str(n_seen) )
all_labels = list(map(restricted_map.get,all_labels))

print("Saving labels")
np.savetxt("../imgs/aug_imgs/aug_labels.dat", all_labels, fmt = "%d")
np.savetxt("../imgs/aug_imgs/aug_vec_labels.dat", all_vec_labels, fmt = "%d")