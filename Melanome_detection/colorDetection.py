import numpy as np
import argparse
import cv2
# load the image
image = cv2.imread("IMGTRAIN/IMD417/IMD417_Dermoscopic_Image/IMD417.bmp")

cv2.imshow("Original Image",image)
def threshFunction(image):
    # Transform to gray colorspace and blur the image.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (17, 17), 32)
    # Perform Otsu threshold.
    ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh
thresh=threshFunction(image)
# Search for contours and iterate over contours. Make threshold for size to
# eliminate others.
im2, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#find max contourArea
cnt = max(contours, key=cv2.contourArea)

if len(cnt) > 4:
        x,y,w,h = cv2.boundingRect(cnt)
        
img2 = image[y:y+h, x:x+w]
thresh2=threshFunction(img2)

img2[thresh2==0] = (0,153,0)
cv2.imshow("Segmented Image",img2)
cv2.waitKey(0);

rgb = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

# define the list of boundaries(white,black,red,blue-gray,darkbrown,lightbrown)
boundaries = [
    ([205, 205, 205], [255, 255, 255]),
    ([0, 0, 0], [50, 50, 50]),
    ([150, 0, 0], [255, 50, 50]),
    ([0, 100, 125], [150, 125, 150]),
    ([50, 0, 0], [150, 100, 100]),
    ([150, 50, 0], [200, 150, 100])
 
]
'''
[
    ([0, 0, 200], [180, 255, 255]),
    ([0, 0, 0], [180, 255, 30]),
    ([0, 100, 100], [10, 255, 255]),
    ([110, 50, 50], [130, 255, 255]),
    ([10, 200, 50], [20, 255, 200]),
    ([10, 100, 50], [20, 200, 200])
 
]
'''
# loop over the boundaries
Values=[]
Colors=['white','black','red','blue-gray','darkbrown','lightbrown']
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower)
    upper = np.array(upper)
 
    # find the colors within the specified boundaries and apply
    # the mask
    
    mask = cv2.inRange(rgb, lower, upper)
    output = cv2.bitwise_and(img2, img2, mask = mask)
    
    if(cv2.countNonZero(mask)<50):
                Values.append(0)
    else:
                Values.append(1)
          
    # show the images
    #cv2.imshow("images", np.hstack([image, output]))
    
    cv2.imshow("Detection",mask)
    cv2.waitKey(0)
    
colorList=[]
for i in range(6):
    if(Values[i]==1):
        colorList.append(Colors[i])
print("list of colors in the image:",colorList)

