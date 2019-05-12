# -*- coding: utf-8 -*-
import imutils
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Read the image and perform threshold and get its height and weight
img = cv2.imread("IMGTRAIN/IMD024.bmp")
h1, w1 = img.shape[:2]
# Transform to gray colorspace and blur the image.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (17, 17), 32)

# Make a fake rectangle arround the image that will seperate the main contour.
cv2.rectangle(blur, (0,0), (w1,h1), (255,255,255), 10)
# Perform Otsu threshold.
ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #result = img.copy()
        #result[thresh!=0] = (0,0,0)

# Create a mask for bitwise operation
mask = np.zeros((h1, w1), np.uint8)
# Search for contours and iterate over contours. Make threshold for size to
# eliminate others.
im2, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#find max contourArea
cnt = max(contours, key=cv2.contourArea)
cv2.drawContours(mask, [cnt] ,-1, 255, -1)
# Perform the bitwise operation.
res = cv2.bitwise_and(thresh, thresh, mask=mask)
#find centroid of main contour
M = cv2.moments(cnt)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

if len(cnt) > 4:
	ellipse = cv2.fitEllipse(cnt)
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.ellipse(img,ellipse,(0,255,0),2)
#draw main contour	
cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)
#draw centroid
cv2.circle(img,(cx,cy),2,(0,0,255),3)
cv2.imshow('original image',img)
cv2.waitKey(0)

#crop image #crop_img = im2[y:y+h, x:x+w]
crop_img = res[y:y+h, x:x+w]

#divise cropped image into 2 equal parts
def diviseImage(img):
    img2 = img
    height, width = img.shape
    # Number of pieces Horizontally 
    CROP_W_SIZE  = 2 
    # Number of pieces Vertically to each Horizontal  
    CROP_H_SIZE = 1
    #create an array to save two parts of main image
    imgArray=[]
    for ih in range(CROP_H_SIZE ):
        for iw in range(CROP_W_SIZE ):

            x = width/CROP_W_SIZE * iw 
            y = height/CROP_H_SIZE * ih
            h = (height / CROP_H_SIZE)
            w = (width / CROP_W_SIZE )
            img = img[y:y+h, x:x+w]
            imgArray.append(img)
            img = img2
    return imgArray

# loop over the rotation angles
scores=[]
for angle in np.arange(0,180,45):
	rotated = imutils.rotate_bound(crop_img, angle)
	#divise into 2 parts
	A = diviseImage(rotated)
	#compare 2 parts
	score = cv2.matchShapes(A[0],A[1],1,0.0)
	scores.append(score)
	print(score)
	#display
	#cv2.imshow("Rotated", rotated)
	#cv2.waitKey(0)
	
print('summary=',sum(scores))
cv2.imshow('cropped image',crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
