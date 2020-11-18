import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image # gives better output control than matplotlib
import sklearn.model_selection as sk
from pyt_models import ConvNet, count_parameters, MyDataset
import pandas as pd
import os 
from tqdm import tqdm

if __name__ == "__main__":
	NORMALISE = False
	torch.set_num_threads(4)

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# device = 'cpu'
	print(device)

	# Hyper parameters
	num_epochs = 30
	batch_size = 250
	learning_rate = 0.001

	print("### Setup ###")

	n_import = 100000
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
	# n_use = 100 # TODO: needed with n_data?
	# print(f"Using {n_use} randomly chosen imgs")
	# ids = np.random.choice(n_data,size=n_use,replace=False)
	# print(ids)
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


	model = ConvNet(im.shape,[5,5],[5,5]).to(device)
	model.train()
	# print(model)

	print(count_parameters(model))
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# print(train_loader)
	# exit(0)
	# Train the model
	print("Begin training:")
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = images.to(device)
			labels = labels.to(device)
			
			# Forward pass
			outputs = model(images)
			
			loss = criterion(outputs[0], labels[:,0])
			for i in range(1,4):
				loss+= criterion(outputs[i], labels[:,i])

			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		
		print(epoch+1, loss.item())

	torch.save(model, '../models/model.ckpt')
	loaders = {'Train' : train_loader,'Test':test_loader}

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


