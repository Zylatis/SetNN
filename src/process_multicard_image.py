import numpy as np
import cv2
import pandas as pd
import copy
import time
import os 
import torch
import matplotlib.pyplot as plt
import classify_card
from PIL import Image, ImageFont, ImageDraw 

# very finicky wrt these area thresholds, threshold list, and the approxPolyDP for rounded corners
MIN_CARD_ARE5A_FRAC = 0.01
MAX_CARD_AREA_FRAC = 0.5
WHITE_CUTOFF = 180  # should really be taking mean across each pixel

def register_cards(image_file = None, image_array = None):
	if image_file is not None:
		file_name = image_file.split("/")[-1].split(".")[0]
		im = cv2.imread( image_file )

	elif image_array is not None:
		im = Image.fromarray(np.uint8(image_array)).convert('RGB')

	else:
		raise ValueError("Must specify either an image path or numpy array in register_cards()")
	im = cv2.resize(im,(500,500),interpolation = cv2.INTER_AREA)
		
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	overlay_im = copy.deepcopy(im)
	threshold_list = [20, 80, 120, 160, 200, 250]
	threshold_list = [ i for i in range(20, 250, 20)]
	detected_cards = [] 

	for t in threshold_list:
		blur = cv2.GaussianBlur(gray, (3,3), 500)
		flag, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
		# thresh =  cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
					# cv2.THRESH_BINARY,3,2)

		contours, hierarchy = cv2.findContours(thresh ,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )

		# cv2.imwrite(f"thresh_{t}.jpg",thresh)
		contours = sorted(contours, key=cv2.contourArea,reverse=True)

		im_w, im_h = im.shape[:2]
		im_area = im_w*im_h
		min_card_area = MIN_CARD_ARE5A_FRAC*im_area
		max_card_area = MAX_CARD_AREA_FRAC*im_area
		card_areas = []

		for i in range(len(contours)):
			card = contours[i]
			area = cv2.contourArea(card)
			
			peri = cv2.arcLength(card,True)
			approx = cv2.approxPolyDP(card, 0.025*peri, True)
			rect = cv2.minAreaRect(card)
			r = cv2.boxPoints(rect).astype(int)
			if len(approx) == 4 and area > min_card_area and area < max_card_area:
				cv2.drawContours(overlay_im,[approx], 0, (0,191,255), 5)
				card_areas.append(area)
				detected_cards.append(approx)

	unique_contours = []
				
	for c1 in detected_cards:
		dupes = 0
		for c2 in unique_contours:
			diff_vec = np.abs(np.sort(c1.flatten())-np.sort(c2.flatten()))
			if (diff_vec <20).all():
				dupes += 1
		if dupes == 0:
			unique_contours.append(c1)

	i = 0

	cards = []

	for el in unique_contours:
		h = np.array([ [0,0],[300,0],[300,300],[0,300] ],np.float32)
		transform = cv2.getPerspectiveTransform(el.astype(np.float32),h)
		warp = cv2.warpPerspective(im,transform,(300,300))
		white_info = warp.mean(axis=2).flatten()
		whitespace_fraction = sum(white_info > 150)/len(white_info)


		# print(np.all(warp > 150, axis=-1).sum()/len(white_info))
		# valid_whitespace = sum(warp.flatten()>WHITE_CUTOFF)/len(warp.flatten())>0.30
		# gs = np.dot(warp[...,:3], [0.299, 0.587, 0.114]).flatten()
		# print(i,sum(gs>WHITE_CUTOFF)/len(gs))
		# np.column_stack(np.where(gray < threshold_level))

		# print(sum(warp.mean(axis=2)>100))
		# print(warp.shape)
		# exit(0)
		# x = np.array(warp)
		# a = np.dot(x.astype(np.uint32),[1,256,65536]) 
		# plt.hist(a) 
		# plt.title("histogram") 
		# plt.savefig(f'hist_{i}.png')
		# plt.clf()
		# # exit(0)
		if whitespace_fraction>0.4:
			# if dump_registered:
				
				# cv2.imwrite(f"../outputs/registered_cards/{file_name}_{i}.jpg",warp)
				# i+=1
			cards.append(warp)
		
	# if dump_registered:
		# cv2.imwrite(f"{file_name}.jpg",overlay_im)



	return cards

if __name__ == '__main__':
	device = 'cpu'
	model = torch.load('../models/model.ckpt', device)
	model.eval()
	files = [
			# 'yt.png',
			'dl1.jpg',
			# 'dl2.jpg',
			# 'dl3.jpg',
			# 'dl4.jpg'
		]
	for f in files:
		# register_cards(f"../data/frames/{f}", dump_registered=True)
		cards = register_cards(image_file = f"../data/frames/{f}")
		i = 0
		for c in cards:
			# r = classify_card.classify_card_array(c, model)
			im = Image.fromarray(c)
			draw = ImageDraw.Draw(im)
			# draw.text((0, 0),  r ,(0,0,0))
			im.save(f'{i}.png')
			# print(r)
			i+=1
	# for i in range(12):
		# print(classify_card.classify_card_file(f'{i}.png', model, shift_axes = True))