import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageFont, ImageDraw # gives better output control than matplotlib
from pyt_models import ConvNet, MyDataset
from classes import get_labels, colours, counts, shape, fill
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def yield_labeled_files(imgs_folder):
	for i in os.listdir( imgs_folder + "processed/"):
		if i.endswith('.png'):
			im = np.asarray(Image.open( imgs_folder + "processed/"+str(i) )).astype(np.uint8)
			yield [im, list(map(str.strip,(i[:-4]).split('_')[2:6])),i ]

def yield_files(imgs_folder):
	for i in os.listdir( imgs_folder + "processed/"):
		if i.endswith('.png'):
			im = np.asarray(Image.open( imgs_folder + "processed/"+str(i) )).astype(np.uint8)
			yield im

def load_data(imgs_folder, has_labels = False, shuffle = False):

	imgs = []
	for el in yield_files(imgs_folder):
		imgs.append(el)

	imgs = np.array(imgs)
	imgs = np.moveaxis(imgs, 3, 1) # assumptions here
	
	dataset = MyDataset(imgs, np.array([0,0,0,0]*len(imgs)))
	data_loader = DataLoader(
		dataset,
		batch_size=len(imgs),
		shuffle=shuffle,
		num_workers=4,
		pin_memory=torch.cuda.is_available()
	)
	return data_loader
	

model = torch.load('../models/model.ckpt', device)#.eval()
model.eval()
# print(model)
loaders = {}
loaders['original'] = load_data("../imgs/")

n=0
for k,loader in loaders.items():
	with torch.no_grad():
		for images, labels in loader:

			images = images.to(device)
			outputs = model(images)
			temp = []
			for i in range(4):
				_, predicted = torch.max(outputs[i], 1)
				temp.append(predicted.tolist())
				
			temp = np.array(temp).transpose()
			predicted_str = []
			for el in temp:
				predicted_str.append("-".join([colours[el[0]],counts[el[1]],fill[el[2]],shape[el[3]]] ))
			
			for i in range(len(images)):
				im = images[i].numpy().astype(np.uint8)
				im = np.moveaxis(im, 0, 2)
				im_show = Image.fromarray(im)

				draw = ImageDraw.Draw(im_show)
				draw.text((0, 0),  predicted_str[i] ,(0,0,0))
				im_show.save( "../imgs/" + "/check_original/" + str(n) + ".png")

				n+=1
				# exit(0)

		# print(f'{k} accuracy:')
		# for i in range(4):
		# 	print(round(100.*correct[i]/total[i],2))


