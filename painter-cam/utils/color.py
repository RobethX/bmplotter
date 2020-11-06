import numpy as np
import cv2 as cv

def BGR2CMYK(img):
    assert img.shape[2] == 3 # make sure the image has 3 channels
    assert img.dtype == np.uint8 # make sure the image is the proper datatype

    old_err_state = np.seterr(divide="ignore", invalid="print") # prevent divide by zero errors

    b, g, r = cv.split(img) # split the BGR image into its channels

    b = np.divide(b, 255., dtype=np.float)
    g = np.divide(g, 255., dtype=np.float)
    r = np.divide(r, 255., dtype=np.float)

    k = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # black "key" channel
    #k = 1 - np.maximum(b, g, r) # black "key" channel
    #k = np.minimum(b, g, r)
    k = np.divide(k, 255., dtype=np.float)
    c = np.divide((1 - r - k), (1 - k)) # cyan channel
    m = np.divide((1 - g - k), (1 - k)) # magenta channel
    y = np.divide((1 - b - k), (1 - k)) # yellow channel

    #c = (1 - r - k) # cyan channel
    #m = (1 - g - k) # magenta channel
    #y = (1 - b - k) # yellow channel

    c = np.uint8(c * 255)
    m = np.uint8(m * 255)
    y = np.uint8(y * 255)
    k = np.uint8(k * 255)

    c *= ~k

    np.seterr(**old_err_state) # restore numpy error handling

    cmyk = cv.merge([m, c, y, k]) # assemble channels
    return cmyk
    
def BGR2CMY(img):
    assert img.shape[2] == 3 # make sure the image has 3 channels
    assert img.dtype == np.uint8 # make sure the image is the proper datatype

    b, g, r = cv.split(img) # split the BGR image into its channels

    c = 255 - r # cyan channel
    m = 255 - g # magenta channel
    y = 255 - b # yellow channel

    cmy = cv.merge([y, m, c]) # assemble channels [c, m, y]
    return cmy

def CMY2BGR(img):
    assert img.shape[2] == 3 # make sure the image has 3 channels
    assert img.dtype == np.uint8 # make sure the image is the proper datatype

    c, m, y = cv.split(img) # split the CMY image into its channels

    b = 255 - c # blue channel
    g = 255 - m # green channel
    r = 255 - y # red channel

    cmy = cv.merge([c, m, y]) # assemble channels
    return cmy

# TODO: use PIL