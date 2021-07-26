# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:37:01 2021

@author: glaub
"""
import cv2
import numpy as np
import imutils

img1 = cv2.imread('ic1.jpg')
img2= cv2.imread('ic2.jpg')
img3 = cv2.imread('ic3.jpg')
img4 = cv2.imread('ic4.jpg')
stitcher = cv2.Stitcher_create()
result = stitcher.stitch((img1,img2, img3, img4))
cv2.imwrite("result.jpg", result[1])
img = cv2.imread('result.jpg') 
#Cropping
img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
mask = np.zeros(thresh.shape, dtype="uint8")
(x, y, w, h) = cv2.boundingRect(c)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
minRect = mask.copy()
sub = mask.copy()
while cv2.countNonZero(sub) > 0:
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)
cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(c)
img = img[y:y + h, x:x + w]
cv2.imwrite("result1.jpg", img)












