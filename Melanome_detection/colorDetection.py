import numpy as np
import argparse
import cv2
# load the image
image = cv2.imread("IMGTRAIN/IMD8.jpg")

hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
cv2.imshow("Original Image",image)
# define the list of boundaries(white,black,red,blue,darkbrown,lightbrown)
boundaries = [
    ([0, 0, 200], [180, 255, 255]),
    ([0, 0, 0], [180, 255, 30]),
    ([0, 100, 100], [10, 255, 255]),
    ([110, 50, 50], [130, 255, 255]),
    ([10, 200, 50], [20, 255, 200]),
    ([10, 100, 50], [20, 200, 200])
 
]
# loop over the boundaries
Values=[]
Colors=['white','black','red','blue','darkbrown','lightbrown']
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower)
    upper = np.array(upper)
 
    # find the colors within the specified boundaries and apply
    # the mask
    
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    
    if(cv2.countNonZero(mask)==0):
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
        
