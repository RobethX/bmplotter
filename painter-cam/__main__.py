import os
import argparse
import cv2 as cv
import numpy as np

from blackstripes import spiral, sketchy, crossed
from penkit_optimize.cli import run_optimizer
import vpype

# Blackstripes
LEVELS = [180, 108, 180, 108]
LINE_WIDTH = 2.0
LINE_SPACING = 2
LINE_COLOR = "#000000"
MAX_LINE_LENGTH = 100
INTERNAL_LINE_SIZE = 2
SCALE = 1.0
ROUND = False

# DEFAULT PARAMETERS
DEF_COLORS = 8

def getOutput(input_path, method=""):
    normpath = os.path.normpath(input_path)
    name = ".".join(normpath.split(".")[:-1])
    if len(method) > 0:
        name += f"-{method}"
    print(name)
    return name + ".svg"


def getOutputPngPath(output_path):
    return output_path + ".png"

def applySpiral():
    draw_args = [
        args.filename,
        getOutput(args.filename, method="spiral"),
        LINE_WIDTH,
        LINE_COLOR,
        SCALE
    ] \
        + LEVELS \
        + [LINE_SPACING] \
        + [0,0,0] \
        + [ROUND]
    spiral.draw(*draw_args)


def applyCrossed():
    draw_args = [
        args.filename,
        getOutput(args.filename, method="crossed"),
        LINE_WIDTH,
        LINE_COLOR,
        SCALE
    ] \
        + LEVELS \
        + [1] \
        + [0,0,0]
    crossed.draw(*draw_args)


def applySketchy():
    draw_args = [
        args.filename,
        getOutput(args.filename, method="sketchy"),
        LINE_WIDTH,
        MAX_LINE_LENGTH,
        LINE_COLOR,
        SCALE,
        INTERNAL_LINE_SIZE,
    ] \
        + [0,0,0]
    sketchy.draw(*draw_args)


def processImage():
    img_raw = cv.imread(args.filename) # load image
    img = img_raw

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv.medianBlur(img,5)
    #img = cv.filter2D(img, -1, kernel)
    img = cv.bilateralFilter(img, 9, 50, 50) # bilateral filter
    #img = cv.ximgproc.anisotropicDiffusion(img, 0.5, 0.02, 10) # 2d anisotropic diffusion

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # convert image to HSV
    hue = hsv[:,:,0] # get hue channel
    saturation = hsv[:,:,1] # get saturation channel
    value = hsv[:,:,2] # get value channel (brightness)

    #res = 256 // args.colors

    #hue = (hue // 32) * 32 # quantize hue

    saturation = cv.normalize(saturation, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    #saturation = (saturation // 32) * 32
    #saturation = cv.adaptiveThreshold(saturation, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    #saturation = cv.bilateralFilter(saturation, 9, 75, 75) # bilateral filter

    value = cv.normalize(value, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    #value = (value // 32) * 32

    #hsv = [hue, saturation, value]
    hsv[:,:,0] = hue
    hsv[:,:,1] = saturation
    hsv[:,:,2] = value

    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR) # convert back to RGB

    edges = cv.Canny(cv.medianBlur(img,3), 40, 60) # detect edges

    img_contours = np.array(img)
    contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # detect contours
    #cv.drawContours(img_contours, contours[0], -1, (0,255,0), 1)

    #mask_blue = cv.inRange(hsv, np.array([55, 0, 0]), np.array([120, 255, 255]))
    #blue = cv.bitwise_and(img, img, mask=mask_blue)
    
    # DEBUG: show image
    #cv.imshow("image", np.hstack([img_raw, img, hsv, img_contours]))
    #cv.imshow("image2", np.hstack([hue, saturation, value, edges]))
    #while cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) == 1:
    #    if cv.waitKey(100) >= 0:
    #        break
    #cv.destroyAllWindows()

    return img
    #return hsv

def main():
    parser = argparse.ArgumentParser(prog="painter-cam")
    parser.add_argument("filename", help="input image to convert")
    parser.add_argument("-c", help="amount of paint colors", action="store", dest="colors", default=DEF_COLORS, type=int)

    global args
    args = parser.parse_args()

    #img = processImage()
    applySpiral()
    applyCrossed()
    applySketchy()

if __name__ == "__main__":
    main()