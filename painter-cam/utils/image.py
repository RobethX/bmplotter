import os
import numpy as np
import cv2 as cv
from PIL import Image

def removeWhite(img, threshold=255, erode_iter=3):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #img_gray_filt = cv.bilateralFilter(img_gray, 9, 50, 50) # bilateral filter
    mask_bool = img_gray >= threshold
    mask = ~(255*mask_bool.astype(np.uint8))
    mask = cv.medianBlur(mask, 5)
    mask = cv.erode(mask, None, iterations=erode_iter)
    img_masked = cv.bitwise_and(img, img, mask=mask)

    return img_masked