import argparse
import cv2 as cv
import numpy as np

# DEFAULT PARAMETERS
DEF_COLORS = 8

def processImage():
    img = cv.imread(args.filename) # load image

    img = cv.bilateralFilter(img, 9, 75, 75)
    #img = cv.anisotropicDiffusion(img, 1,0, 0.02, 8)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # convert image to HSV
    hue = hsv[:,:,0] # get hue channel
    saturation = hsv[:,:,1] # get saturation channel
    value = hsv[:,:,2] # get value channel (brightness)

    hue = (hue // 32) * 32 # quantize hue
    #saturation = (saturation // 128) * 128
    #value = (value // 128) * 128

    #hsv = [hue, saturation, value]
    hsv[:,:,0] = hue
    hsv[:,:,1] = saturation
    hsv[:,:,2] = value

    img_processed = cv.cvtColor(hsv, cv.COLOR_HSV2BGR) # convert back to RGB
    return img_processed

def main():
    parser = argparse.ArgumentParser(prog="painter-cam")
    parser.add_argument("filename", help="input image to convert")
    parser.add_argument("-c", help="amount of paint colors", action="store", dest="colors", default=DEF_COLORS, type=int)

    global args
    args = parser.parse_args()

    img = processImage()

    cv.imshow("res", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()