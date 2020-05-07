import numpy as np
import cv2
import pandas as pd
import copy

# very finicky wrt these area thresholds, threshold list, and the approxPolyDP for rounded corners
MIN_CARD_ARE5A_FRAC = 0.002
MAX_CARD_AREA_FRAC = 0.8

def register_cards(file, dump_registered = False):
	file_name = file.split("/")[-1]

	im = cv2.imread( file)
	im = cv2.resize(im,(500,500),interpolation = cv2.INTER_AREA)
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	overlay_im = copy.deepcopy(im)
	threshold_list = [20,80,120, 160,200,250]
	detected_cards = []

	for t in threshold_list:
		blur = cv2.GaussianBlur(gray, (3,3), 500)
		flag, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
		# thresh =  cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		            # cv2.THRESH_BINARY,3,2)

		contours, hierarchy = cv2.findContours(thresh ,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )

		cv2.imwrite(f"thresh_{t}.jpg",thresh)
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
		h = np.array([ [0,0],[128,0],[128,128],[0,128] ],np.float32)
		transform = cv2.getPerspectiveTransform(el.astype(np.float32),h)
		warp = cv2.warpPerspective(im,transform,(128,128))
		if dump_registered:

			cv2.imwrite(f"registered_cards/{file_name}_{i}.jpg",warp)
		cards.append(warp)
		i+=1
	if dump_registered:
		cv2.imwrite(f"{file_name}.jpg",overlay_im)

	return cards

if __name__ == '__main__':
	files = ["img_1963-e1371071035156.jpg",
			"setgame1.jpg",
			"set-game-5-360x240.jpg",
			"SetCards2.jpg",
			"setgame11small.jpg",
			"hand.png",
			"687474703a2f2f6935382e74696e797069632e636f6d2f326e31726763392e706e67.png",
			"set_solid.png"	
		]
	register_cards(f"../imgs/raw/{files[1]}")
