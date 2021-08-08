# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:07:04 2021

@author: glaub
"""
from blur_detector import detect_blur_fft
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2 as cv
orig = cv.imread("blur.jpg")
orig = imutils.resize(orig, width=500)
gray = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
(mean, blurry) = detect_blur_fft(gray, size=60)
image = np.dstack([gray] * 3)
color = (0, 0, 255) if blurry else (0, 255, 0)
text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
text = text.format(mean)
cv.putText(image, text, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7,
	color, 2)
print("[INFO] {}".format(text))
plt.subplot(111),plt.imshow(image)
plt.title('Image'), plt.xticks([]), plt.yticks([])
plt.show()



