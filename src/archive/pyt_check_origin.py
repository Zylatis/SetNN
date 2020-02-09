import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageFont, ImageDraw # gives better output control than matplotlib
from pyt_models import ConvNet, MyDataset
from classes import get_labels, colours, counts, shape, fill
import os


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
model.eval()
# print(model)
label_lookup = get_labels()[1]
original_files = yield_files()
imgs = []
labels = []
labels_str = {}

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
imgs = np.moveaxis(imgs, 3, 1)

labels = np.array(labels)
# print(labels)
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
# font = ImageFont.truetype(arial.t, 12)

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
			# print(wrong.flatten())
			temp = np.array(temp).transpose()

			for el in temp:
				predicted_str.append("-".join([colours[el[0]],counts[el[1]],fill[el[2]],shape[el[3]]] ))
			
			for i in range(len(images)):
				im = images[i].numpy().astype(np.uint8)
				im = np.moveaxis(im, 0, 2)
				im_show = Image.fromarray(im)
				# print(predicted_str[i])
				draw = ImageDraw.Draw(im_show)
				draw.text((0, 0),  predicted_str[i] ,(255,255,255))
				im_show.save( imgs_folder + "/check_original/" + str(n) + ".png")

				n+=1
				# exit(0)

		print(f'{k} accuracy:')
		for i in range(4):
			print(round(100.*correct[i]/total[i],2))


