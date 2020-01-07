import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFont, ImageDraw # gives better output control than matplotlib

from pyt_models import ConvNet, MyDataset
import os
from classes import get_labels, colours, counts, shape, fill
imgs_folder = "../imgs/"
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def yield_files():
	for i in os.listdir( imgs_folder + "processed/"):
		if i.endswith('.png'):
			# label =  ("_").join(  (i[:-4]).split('_')[2:6] ).strip()
			im = np.asarray(Image.open( imgs_folder + "processed/"+str(i) )).astype(np.uint8)
			yield [im, list(map(str.strip,(i[:-4]).split('_')[2:6])),i ]

model = torch.load('../models/model.ckpt', device)
# print(model)
label_lookup = get_labels()[1]
original_files = yield_files()
imgs = []
labels = []
labels_str = {}

# colours = ['red', 'purple', 'green']
# counts = ['single','double', 'triple']
# shape = ['pill', 'diamond', 'squiggle']
# fill = ['empty', 'grid', 'solid']

for el in original_files:
	imgs.append(el[0])
	label = [
		colours.index(el[1][0]),
		counts.index(el[1][1]),
		fill.index(el[1][2]),
		shape.index(el[1][3])	
	]

	labels.append(label)

imgs = np.array(imgs)

# test = Image.fromarray(imgs[0])
# print(imgs[0].shape)
# test.save( imgs_folder + "/check_original/" + str(0) + ".png")
# exit(0)
imgs = np.moveaxis(imgs, 3, 1)

labels = np.array(labels)

dataset = MyDataset(imgs, labels)
data_loader = DataLoader(
	dataset,
	batch_size=len(imgs),
	shuffle=True,
	num_workers=4,
	pin_memory=torch.cuda.is_available()
)
loaders = {'Originals':data_loader}
n=0

for k,loader in loaders.items():
	with torch.no_grad():
		correct = np.zeros(4)
		total = np.zeros(4)
		for images, labels in loader:
			images = images.to(device)
			# labels = labels.to(device)
			outputs = model(images)
			temp = []
			predicted_str = []
			wrong = np.array([])
			for i in range(4):
				_, predicted = torch.max(outputs[i], 1)
				temp.append(predicted.tolist())
				total[i] += labels.size(0)
				correct[i] += (predicted == labels[:,i]).sum().item()
				tt = (predicted == labels[:,i]).numpy()
				# wrong.append(np.where(tt == False)[0])
				wrong = np.append(wrong,np.where(tt == False)[0])
				# exit(0)
			print(wrong.flatten())
			temp = np.array(temp).transpose()

			for el in temp:
				predicted_str.append("-".join([colours[el[0]],counts[el[1]],fill[el[2]],shape[el[3]]] ))
			
			for i in range(len(images)):
				im = images[i].numpy().astype(np.uint8)
				im = np.moveaxis(im, 0, 2)
				im_show = Image.fromarray(im)

				draw = ImageDraw.Draw(im_show)
				draw.text((0, 0),  predicted_str[i] ,(0,0,0))
				im_show.save( imgs_folder + "/check_original/" + str(n) + ".png")

				n+=1
				# exit(0)

		print(f'{k} accuracy:')
		for i in range(4):
			print(round(100.*correct[i]/total[i],2))


