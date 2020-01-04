import numpy as np
import cv2

im = cv2.imread( "../imgs/raw/setgame11small.jpg")
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=0)

blur = cv2.GaussianBlur(gray,(1,1),1000)
flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
numcards=13
contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards]  

for i in range(numcards):
	card = contours[i]
	peri = cv2.arcLength(card,True)
	approx = cv2.approxPolyDP(card,0.01*peri,True)
	rect = cv2.minAreaRect(card)
	r = cv2.boxPoints(rect).astype(int)
	cv2.drawContours(im,[approx],0,(0,191,255),2)

h = np.array([ [0,0],[500,0],[500,500],[0,500] ],np.float32)
transform = cv2.getPerspectiveTransform(approx.astype(np.float32),h)
warp = cv2.warpPerspective(im,transform,(500,500))
cv2.imwrite("test.jpg",im)
cv2.imwrite("test2.jpg",warp)
