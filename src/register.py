import numpy as np
import cv2
import pandas as pd
import copy
# file = "img_1963-e1371071035156.jpg"
# file = "setgame1.jpg"
# file = "set-game-5-360x240.jpg"
# file = "SetCards2.jpg"
# file = "setgame11small.jpg"
file = "hand.png"
# file = "687474703a2f2f6935382e74696e797069632e636f6d2f326e31726763392e706e67.png"
# file = "set_solid.png"	
im = cv2.imread( f"../imgs/raw/{file}")
im = cv2.resize(im,(500,500),interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

threshold_list = [100,120,160,200]
detected_cards = []

for t in threshold_list:
	blur = cv2.GaussianBlur(gray,(3,3),500)
	flag, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
	# thresh =  cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#             cv2.THRESH_BINARY,3,2)

	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.imwrite("thresh.jpg",thresh)
	contours = sorted(contours, key=cv2.contourArea,reverse=True)

	im_w, im_h = im.shape[:2]
	im_area = im_w*im_h
	MIN_CARD_AREA = 0.01*im_area
	MAX_CARD_AREA = 0.95*im_area
	card_areas = []
	for i in range(len(contours)):
		card = contours[i]
		area = cv2.contourArea(card)
		
		peri = cv2.arcLength(card,True)

		approx = cv2.approxPolyDP(card,0.02*peri,True)
		rect = cv2.minAreaRect(card)
		r = cv2.boxPoints(rect).astype(int)

		if len(approx) ==4 and area > MIN_CARD_AREA and area < MAX_CARD_AREA:
			cv2.drawContours(im,[approx],0,(0,191,255),5)
			card_areas.append(area)
			detected_cards.append(approx)


unique_contours = []
# for i in range(len(detected_cards)):
# 	dupes = 0
# 	c1 = detected_cards[i]
# 	for j in range(i,len(detected_cards)):
# 		c2 = detected_cards[j]
# 		if (c1 != c2).all():
# 			diff_vec = np.abs(np.sort(c1.flatten())-np.sort(c2.flatten()))
# 			if (diff_vec <30).all():
# 				detected_cards[j] = copy.deepcopy(detected_cards[i])
			
for c1 in detected_cards:
	dupes = 0
	for c2 in unique_contours:
		diff_vec = np.abs(np.sort(c1.flatten())-np.sort(c2.flatten()))
		if (diff_vec <20).all():
			dupes += 1
	if dupes == 0:
		unique_contours.append(c1)


# unique_contours = np.unique(detected_cards, axis=0)

print(np.sort(unique_contours[1].flatten()))
print(np.sort(unique_contours[2].flatten()))
# exit(0)
i=0
for el in unique_contours:
	h = np.array([ [0,0],[128,0],[128,128],[0,128] ],np.float32)
	transform = cv2.getPerspectiveTransform(el.astype(np.float32),h)
	warp = cv2.warpPerspective(im,transform,(128,128))
	cv2.imwrite(f"registered_cards/{i}.jpg",warp)
	i+=1

cv2.imwrite("test.jpg",im)
