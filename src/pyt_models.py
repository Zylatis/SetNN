import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import fns
import numpy as np
from PIL import Image # gives better output control than matplotlib
import sklearn.model_selection as sk


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50
batch_size = 100
learning_rate = 0.001



# model = "vec_"
model= ""
print("### Setup ###")

imgs_folder = "../imgs/aug_imgs/"
# labels = np.loadtxt( "../imgs/aug_imgs/aug_"+model+"labels.dat").astype(np.int32)
vec_labels = np.asarray(np.loadtxt( "../imgs/aug_imgs/aug_vec_labels.dat").astype(np.int32))

# colour, count, shape, fill
n_data = len(vec_labels)
n_data = 10
print("Loading " + str( n_data ) + " images: "),

imgs = []
for i in range(n_data):
	im = np.asarray(Image.open( imgs_folder +str(i) + '.png' ))#.convert('LA'))
	imgs.append(im)
	img_mean = int(np.mean(im.flatten()))
	# im = im - img_mean
	# im_show = Image.fromarray(im)
	# im_show.show()
	# exit(0)
	
print(im.shape)
print("Done.")
# random.seed(0)
imgs = np.asarray(imgs)

# Here we move the channel index around to match the pytorch requirement
imgs = np.moveaxis(imgs, 3, 1)
n = len( imgs )
img_train, img_test, class_train, class_test = sk.train_test_split( imgs[:n], vec_labels[:n], test_size = 0.1, shuffle = True )

img_train = img_train
img_test = img_test
class_train = class_train
class_test = class_test


dataset = MyDataset(img_train, class_train)
loader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
	def __init__(self,dims, kernels, filters):
		super(ConvNet, self).__init__()
		self.dims = dims
		self.kernels=kernels
		self.filters = filters
		
		# self.layers = [0,0,0,0]
		self.pos = {"colour" : 0, "counts" : 1, "fill" : 2, "shape" : 3}

		self.branch("colour")
		self.branch("counts")
		self.branch("fill")
		self.branch("shape")

		# self.myparameters = nn.ParameterList(self.layers[0][0])

	def branch(self, name):
		layer1 = nn.Sequential(
			nn.Conv2d(self.dims[2], self.filters[0], kernel_size=self.kernels[0], stride=1, padding=2),
			nn.BatchNorm2d(self.filters[0]),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))

		layer2 = nn.Sequential(
			nn.Conv2d(self.filters[0], self.filters[1], kernel_size=self.kernels[1], stride=1, padding=2),
			nn.BatchNorm2d(self.filters[1]),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))

		fc = nn.Linear(int(self.dims[0]/4*self.dims[1]/4*self.filters[1]), 4) # each branch only gives one class

		# Rather than storing things in lists we assign to the class with the nice add_module fn
		# so it picks up the parameter sets 
		self.add_module(f"{name}_layer1", layer1)
		self.add_module(f"{name}_layer2", layer2)
		self.add_module(f"{name}_fc", fc)

	def forward(self, x):
		output_set = [-1,-1,-1,-1]
		layers = model._modules
		for k in self.pos.keys():

			out = layers[f"{k}_layer1"](x)
			out = layers[f"{k}_layer2"](out)
			out = out.reshape(out.size(0), -1)
			out = layers[f"{k}_fc"](out)
			output_set[self.pos[k]] = out 
		return output_set

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
model = ConvNet(im.shape,[5,5],[5,5]).to(device)


print(count_parameters(model))
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(loader):
		images = images.to(device)
		labels = labels.to(device)
		
		# Forward pass
		outputs = model(images)
		
		loss = criterion(outputs[0], labels[:,0]) 
		# for i in range(1,4):
			# loss +=  criterion(outputs[i], labels[:,i])  
		# print(outputs[1])
		# exit(0)
		# # Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# exit(0)
		print(loss.item())
		print(outputs[0][0])
		print(labels[0])