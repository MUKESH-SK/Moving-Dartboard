import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math

# initialise the Video
cap =cv2.VideoCapture('Videos/vid (1).mp4')
# create the color finder object
myColorFinder = ColorFinder(True)
hsvVals = {'hmin':8,'smin':124,'vmin':13,'hmax':24,'smax':255,'v:max':255,}
while True:
    # Grab the image
    # success,im
    img = cv2.imread("ball4.png")
    img = img[0:900, :]
    # Find the color ball
    imgColor,mask = myColorFinder.update(img,hsvVals)

    # Display
    img = cv2.resize(img,(0,0),None,0.7,0.7)
    # cv2.imshow("Image",img)
    cv2.imshow("ImageColor",imgColor)
    cv2.waitKey(50)