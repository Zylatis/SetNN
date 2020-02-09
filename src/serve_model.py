import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageFont, ImageDraw 
from pyt_models import ConvNet, MyDataset
from classes import get_labels, class_labels
import os
import cv2
from register import register_cards
from resize import resize_img

input_size = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def yield_labeled_files(imgs_folder, resize = True):
	for i in os.listdir( imgs_folder ):
		if i.endswith('.png'):
			im = Image.open( imgs_folder + str(i) )
			if resize:
				im = resize_img(im, input_size)
			yield [np.asarray(im).astype(np.uint8), np.asarray(list(map(str.strip,(i[:-4]).split('_')[2:6]))),i ]

def yield_files(imgs_folder, resize = True):
	for i in os.listdir( imgs_folder ):
		if i.endswith('.png'):
			im = Image.open( imgs_folder + str(i) )
			if resize:
				im = resize_img(im, input_size)
			yield 

def load_data(imgs_folder, has_labels = False, shuffle = False):

	imgs = []
	labels = []
	if has_labels:
		for el in yield_labeled_files(imgs_folder):
			imgs.append(el[0])
			label = [
				class_labels['colours'].index(el[1][0]),
				class_labels['counts'].index(el[1][1]),
				class_labels['fill'].index(el[1][2]),
				class_labels['shape'].index(el[1][3])	
			]
			labels.append(np.array(label))

	else:
		for el in yield_files(imgs_folder):
			imgs.append(el)
		labels = np.array([0,0,0,0]*len(imgs))

	imgs = np.array(imgs)
	labels = np.array(labels)
	imgs = np.moveaxis(imgs, 3, 1) # assumptions here
	
	dataset = MyDataset(imgs, labels)
	data_loader = DataLoader(
		dataset,
		batch_size=len(imgs),
		shuffle=shuffle,
		num_workers=4,
		pin_memory=torch.cuda.is_available()
	)

	return data_loader

def label_images(images, labels, model, dump_folder = None):
	images = images.to(device)
	labels = labels.to(device)
	outputs = model(images)
	temp = []

	correct = np.zeros(4)
	total = np.zeros(4)
	for i in range(4):
		_, predicted = torch.max(outputs[i], 1)
		temp.append(predicted.tolist())
		total[i] += labels.size(0)
		correct[i] += (predicted == labels[:,i]).sum().item()
		# exit(0)
		
	temp = np.array(temp).transpose()
	predicted_str = []
	for el in temp:
		predicted_str.append("-".join([
			class_labels['colours'][el[0]],
			class_labels['counts'][el[1]],
			class_labels['fill'][el[2]],
			class_labels['shape'][el[3]]
		]))
	
	for i in range(len(images)):
		print(i)
		im = images[i].numpy().astype(np.uint8)
		im = np.moveaxis(im, 0, 2)
		im_show = Image.fromarray(im)

		draw = ImageDraw.Draw(im_show)
		draw.text((0, 0),  predicted_str[i] ,(255,255,255))
		if dump_folder != None:
			im_show.save(f"{dump_folder}/{i}.png") #

	print(f'Accuracy:')
	for i in range(4):
		print(round(100.*correct[i]/total[i],2))
		
def check_batch(img_folder, model, has_labels = False):
	
	loader = load_data(img_folder, has_labels)

	with torch.no_grad():
		for images, labels in loader:
			label_images(images, labels, model, f"{img_folder}labeled/")
			
def check_frame(path, model, labels = []):
	cards = register_cards(path)

	for card in cards:
		card = np.moveaxis(card, 2, 0)
		t = torch.from_numpy(card).unsqueeze(0).float()
		outputs = model(t)
		temp = []
		for i in range(4):
			_, predicted = torch.max(outputs[i], 1)
			temp.append(predicted.tolist())
			
		temp = np.array(temp).transpose()
		print(temp)
		exit(0)
	model(im)

model = torch.load('../models/model.ckpt', device)
model.eval()


check_batch("../imgs/processed/", model, True)
# check_frame("../imgs/raw/img_1963-e1371071035156.jpg",model)
