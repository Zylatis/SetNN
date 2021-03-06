import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

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


# CNN (two convolutional layers)
class ConvNet(nn.Module):
	def __init__(self,dims, filters, kernel_sizes):
		super(ConvNet, self).__init__()
		self.dims = dims
		self.kernel_sizes = kernel_sizes
		self.filters = filters
		
		self.pos = {"colour" : 0, "counts" : 1, "fill" : 2, "shape" : 3}

		self.branch("colour")
		self.branch("counts")
		self.branch("fill")
		self.branch("shape")

		# determine size of FC needed
		layers = self._modules
		x = np.random.rand(*dims)
		x = np.moveaxis(x, 2, 0)
		x = torch.from_numpy(x).unsqueeze(0).float()

		x = layers[f"colour_layer1"](x)
		x = layers[f"colour_layer2"](x)
		for k in self.pos.keys():
			self.add_fc((x.reshape(-1)).shape[0], k)

	def branch(self, name):
		layer1 = nn.Sequential(
			nn.Conv2d(self.dims[2], self.filters[0], kernel_size=self.kernel_sizes[0], stride=1, padding=2),
			nn.BatchNorm2d(self.filters[0]),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		
		layer2 = nn.Sequential(
			nn.Conv2d(self.filters[0], self.filters[1], kernel_size=self.kernel_sizes[1], stride=1, padding=2),
			nn.BatchNorm2d(self.filters[1]),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
			)
		

		# Rather than storing things in lists we assign to the class with the nice add_module fn
		# so it picks up the parameter sets 
		self.add_module(f"{name}_layer1", layer1)
		self.add_module(f"{name}_layer2", layer2)
		
	def add_fc(self,size, name):
		fc = nn.Linear(int(size), 4) # each branch only gives one class
		self.add_module(f"{name}_fc", fc)
	
	def forward(self, x):
		output_set = [-1,-1,-1,-1]
		layers = self._modules
		for k in self.pos.keys():
			out = layers[f"{k}_layer1"](x)
			out = layers[f"{k}_layer2"](out)
			out = out.reshape(out.size(0), -1)
			out = layers[f"{k}_fc"](out)
			output_set[self.pos[k]] = out 
		return output_set

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)