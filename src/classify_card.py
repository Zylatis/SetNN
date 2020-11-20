import os
import torch 
import numpy as np
import time
import torch.nn as nn

from PIL import Image, ImageFont, ImageDraw 
from classes import class_labels
from image_processing import resize_img
from pprint import pprint

MODEL_INPUT_SIZE = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
classes = class_labels

def classify_card_array(card_array, model, shift_axes = True):
	if shift_axes:
		card_array = np.moveaxis(card_array, 2, 0)
	
	card_array_torch = torch.from_numpy(card_array).unsqueeze(0).float().to(device)
	outputs = model(card_array_torch)

	# Pretty janky, could make a lot better but requires revisiting training code
	card_classes = [torch.max(outputs[i], 1)[1].tolist()[0] for i in range(4)]

	card_class_string = " ".join([ list(class_labels.values())[i][card_classes[i]] for i in range(4)])
	return card_class_string

def classify_card_file(card_file, model, shift_axes = True):
	im = Image.open( card_file )
	im = resize_img(im, MODEL_INPUT_SIZE)
	im_array = np.asarray(im).astype(np.uint8)
	
	class_str = classify_card_array(im_array, model, shift_axes)
	
	return class_str

if __name__ == '__main__':
	model = torch.load('../models/model.ckpt', device)
	model.eval()
	eval_folder = '../data/train/raw/'
	# eval_folder = '../outputs/registered_cards/'


	# correct = 0
	# n = 0
	# for f in os.listdir(eval_folder):

	# 	if f.endswith('.png'):
	# 		n+=1
	# 		class_text = ' '.join(f.split(".png")[0].split("_")[:4])
	# 		r = classify_card_file(f'{eval_folder}{f}', model)
	# 		if r != class_text:
	# 			print(f)
	# 			print(r)
	# 			print("-"*50)
	# 			print(correct/n)
	# 		else:
	# 			correct+=1
	# print(correct/n)
		
	print(classify_card_file("../imgs/green_double_filled_squiggle.png", model))
	print(classify_card_file("../imgs/yt2.png", model))
	print(classify_card_file("../imgs/yt2p2.png", model))

	print(classify_card_file("../imgs/6.png", model))
	# print(classify_card_file("6.png", model))
	print(classify_card_file("../imgs/6p2.png", model))
	print(classify_card_file("../imgs/6p3.png", model))

	print(classify_card_file("../imgs/test.PNG", model))
	print(classify_card_file("../imgs/test2.PNG", model))
	print(classify_card_file("../imgs/single.PNG", model))
