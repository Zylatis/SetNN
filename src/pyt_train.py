import os 
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import classes
import sklearn.model_selection as sk
from tqdm import tqdm
from pyt_models import ConvNet, count_parameters, MyDataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image # gives better output control than matplotlib

if __name__ == "__main__":
	NORMALISE = False
	torch.set_num_threads(4)


	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print(f'Using device: {device}')

	# Hyper parameters
	num_epochs = 200
	batch_size = 128
	learning_rate = 0.0001

	print("### Setup ###")

	n_import = 100000000
	imgs_folder = "../data/train/augmented/"
	label_df = pd.read_csv( f"{imgs_folder}aug_vec_labels.csv")[:n_import] # to int32, also refactor to remove need for pandas
	label_df.columns = ['file','colour','count','fill','shape']

	print("Loading " + str( len(label_df) ) + " images: "),
	imgs = []
	# For now pull all images into memory
	for el in tqdm(label_df.to_dict('records')[:n_import]):
		im = np.asarray(Image.open( imgs_folder + el['file'] ))#.convert('LA'))
		img_mean = int(np.mean(im.flatten()))
		if NORMALISE:
			im = im - img_mean

		imgs.append(im)

	print(im.shape)
	print("Done.")
	imgs = np.asarray(imgs)

	# Here we move the channel index around to match the pytorch requirement
	imgs = np.moveaxis(imgs, 3, 1)
	vec_labels = label_df[['colour','count','fill','shape']].values
	
	img_train, img_test, class_train, class_test = sk.train_test_split( imgs, vec_labels, test_size = 0.1, shuffle = True )
	img_train = img_train
	img_test = img_test
	class_train = class_train
	class_test = class_test

	train_dataset = MyDataset(img_train, class_train)
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=6,
		pin_memory=torch.cuda.is_available()
	)

	test_dataset = MyDataset(img_test, class_test)
	test_loader = DataLoader(
		test_dataset,
		batch_size=len(class_test),
		shuffle=True,
		num_workers=6,
		pin_memory=torch.cuda.is_available()
	)


	model = ConvNet(im.shape,[6,12],[6,12]).to(device)
	# model = ConvNet(im.shape,[2,2],[2,2]).to(device)

	model.train()

	print(count_parameters(model))
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	L2 = 0.00
	L1 = 0.00
	
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = L2)

	loss_scale = [1,1,1,1]
	# Train the model
	print("Begin training:")
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = images.to(device)
			labels = labels.to(device)
			
			# Forward pass
			outputs = model(images)
			regularization_loss = 0

			for param in model.parameters():
				regularization_loss += L1*torch.sum(torch.abs(param))
			loss = 0  + regularization_loss

			for i in range(4):
				l = criterion(outputs[i], labels[:,i])*loss_scale[i]
			
				# print(f"Property {list(classes.class_labels)[i]} has loss {l}")
				loss+= l
			# print("-"*100)

			# Backward and optimize
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
			
		
		print(epoch+1, loss.item())
		torch.save(model, f'../models/model_{epoch}.ckpt')

	torch.save(model, '../models/model.ckpt')
	loaders = {'Train' : train_loader,'Test':test_loader}
	device = 'cpu'
	model.eval()
	model.to(device)
	for k,loader in loaders.items():
		with torch.no_grad():
			correct = np.zeros(4)
			total = np.zeros(4)
			for images, labels in loader:
				images = images.to(device)
				labels = labels.to(device)
				outputs = model(images)
				for i in range(4):
					_, predicted = torch.max(outputs[i], 1)
					total[i] += labels.size(0)
					correct[i] += (predicted == labels[:,i]).sum().item()

			print(f'{k} accuracy:')
			for i in range(4):
				print(round(100.*correct[i]/total[i],2))


