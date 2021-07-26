# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:03:31 2021

@author: glaub
"""

from PIL import Image
import numpy
import matplotlib.pyplot as plt
import cv2


#%% CROP 

image = Image.open('katyperry.png')
image_arr = numpy.array(image)
image_arr = image_arr[90:200, 200:300]
image = Image.fromarray(image_arr)
plt.imshow(image)
plt.show()


#%% FLIP

image_arr = numpy.array(image)
flip = numpy.flipud(image_arr)
image = Image.fromarray(flip)
plt.imshow(image)
plt.show()

#%% TRANSLATION

image = cv2.imread('katyperry.png')
height, width = image.shape[:2]
quarter_height, quarter_width = height / 4, width / 4
T = numpy.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
img_translation = cv2.warpAffine(image, T, (width, height))
plt.imshow(img_translation)
plt.show()


#%% ROTATION

image = cv2.imread('katyperry.png')
img_rotation = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(img_rotation)
plt.show()

#%% RESIZE 

image = cv2.imread('katyperry.png')
width = int(image.shape[1] * 50 / 100)
height = int(image.shape[0] * 50 / 100)
dsize = (width, height)
image_resize = cv2.resize(image, dsize)
plt.imshow(image_resize)
plt.show()


#%% BITWISE

img1 = cv2.imread("katyperry.png")
img2 = cv2.imread("orlando.jpg")
bitwise_and = cv2.bitwise_and(img2, img1)
plt.imshow(bitwise_and)
plt.show()

bitwise_or = cv2.bitwise_or(img2, img1)
plt.imshow(bitwise_or)
plt.show()

bitwise_xor = cv2.bitwise_xor(img2, img1)
plt.imshow(bitwise_xor)
plt.show()









