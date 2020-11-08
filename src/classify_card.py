import torch 
import numpy as np
from PIL import Image, ImageFont, ImageDraw 
from classes import class_labels
from resize import resize_img


MODEL_INPUT_SIZE = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = class_labels

def classify_card_array(card_array, model, shift_axes = True):
	if shift_axes:
		card_array = np.moveaxis(card_array, 2, 0)
	
	card_array = torch.from_numpy(card_array).unsqueeze(0).float()
	outputs = model(card_array)

	# Pretty janky, could make a lot better but requires revisiting training code
	card_classes = [torch.max(outputs[i], 1)[1].tolist()[0] for i in range(4)]

	card_class_string = " ".join([ list(class_labels.values())[i][card_classes[i]] for i in range(4)])
	return card_class_string

def classify_card_file(card_file, model, shift_axes = True):
	im = Image.open( card_file )
	im = resize_img(im, MODEL_INPUT_SIZE)
	im_array = np.asarray(im).astype(np.uint8)
	return classify_card_array(im_array, model, shift_axes)



model = torch.load('../models/model.ckpt', device)
model.eval()
r = classify_card_file('green_double_empty_squiggle.png', model)
print(r)