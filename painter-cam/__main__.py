import os
import argparse
import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt

from blackstripes import spiral, sketchy, crossed
from svgpathtools import svg2paths, wsvg
import vpype
from hatched import hatched

from utils.color import *
from utils.path import TempFolder
from generators.Squiggle import Squiggle

NUM_SHADES = 4 # number of shading levels
#COLORS = ["#FF0000", "#00FF00", "#0000FF"] # TODO: svgpathtools has a utility for converting hex colors
#COLORS = ["#00FFFF", "#FF00FF", "#FFFF00", "#000000"] # CMYK
COLORS = ["#FFFF00", "#FF00FF", "#00FFFF"] # YMC

# Blackstripes
DEF_LEVELS = [200, 146, 110, 56] # TODO: generate with k-means
DEF_LINE_COLOR = "#000000"
LINE_WIDTH = 0.5
LINE_SPACING = 1
MAX_LINE_LENGTH = 100
INTERNAL_LINE_SIZE = 2
SCALE = 1.0
ROUND = False
SIG_TRANSFORM = [0, 0, 0]

# K-Means parameters
KMEANS_ACCURACY = 0.85 # percent
KMEANS_ITERATIONS = NUM_SHADES
KMEANS_ATTEMPTS = 100
KMEANS_FLAGS = cv.KMEANS_RANDOM_CENTERS
KMEANS_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# DEFAULT PARAMETERS
DEF_NUM_COLORS = 8

def getOutput(input_path, method=""):
    normpath = os.path.normpath(input_path)
    name = ".".join(normpath.split(".")[:-1])
    if len(method) > 0:
        name += f"-{method}"
    #print(name)
    return name + ".svg"

def getOutputPngPath(output_path):
    return output_path + ".png"

def applySpiral(input_path, line_color=DEF_LINE_COLOR, levels=DEF_LEVELS):
    output_path = getOutput(input_path, method="spiral")
    draw_args = [
        input_path,
        output_path,
        LINE_WIDTH,
        line_color,
        SCALE
    ] \
        + levels \
        + [LINE_SPACING] \
        + SIG_TRANSFORM \
        + [ROUND]
    spiral.draw(*draw_args)
    return output_path


def applyCrossed(input_path, line_color=DEF_LINE_COLOR, levels=DEF_LEVELS):
    output_path = getOutput(input_path, method="crossed")
    draw_args = [
        input_path,
        output_path,
        LINE_WIDTH,
        line_color,
        SCALE
    ] \
        + levels \
        + [1] \
        + SIG_TRANSFORM
    crossed.draw(*draw_args)
    return output_path


def applySketchy(input_path, line_color=DEF_LINE_COLOR):
    output_path = getOutput(input_path, method="sketchy")
    draw_args = [
        input_path,
        output_path,
        LINE_WIDTH,
        MAX_LINE_LENGTH,
        line_color,
        SCALE,
        INTERNAL_LINE_SIZE,
    ] \
        + SIG_TRANSFORM
    sketchy.draw(*draw_args)
    return output_path


def processImage():
    img_raw = cv.imread(args.filename) # load image
    img = img_raw

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #img = cv.medianBlur(img,5)
    #img = cv.filter2D(img, -1, kernel)
    img = cv.bilateralFilter(img, 9, 50, 50) # bilateral filter
    #img = cv.ximgproc.anisotropicDiffusion(img, 0.5, 0.02, 10) # 2d anisotropic diffusion

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # convert image to HSV
    hue = hsv[:,:,0] # get hue channel
    saturation = hsv[:,:,1] # get saturation channel
    value = hsv[:,:,2] # get value channel (brightness)

    #res = 256 // args.colors

    #hue = (hue // 32) * 32 # quantize hue

    #saturation = cv.normalize(saturation, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    #saturation = (saturation // 32) * 32
    #saturation = cv.adaptiveThreshold(saturation, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    #saturation = cv.bilateralFilter(saturation, 9, 75, 75) # bilateral filter

    #value = cv.normalize(value, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
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
    cv.imshow("image", np.hstack([img_raw, img, hsv, img_contours]))
    #cv.imshow("image2", np.hstack([hue, saturation, value, edges]))
    while cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) == 1:
        if cv.waitKey(100) >= 0:
            break
    cv.destroyAllWindows()

    return img
    #return hsv

def main():
    parser = argparse.ArgumentParser(prog="painter-cam")
    parser.add_argument("filename", help="input image to convert")
    parser.add_argument("-c", help="amount of paint colors", action="store", dest="colors", default=DEF_NUM_COLORS, type=int)

    global args
    args = parser.parse_args()

    # TODO: check if file exists
    #img = processImage()
    img = cv.imread(args.filename)

    tmp = TempFolder()

    tmp_path = os.path.normpath("tmp")
    #tmp_path = tmp.getPath()
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    #cmyk = BGR2CMYK(img)
    #cmy = BGR2CMY(img)

    channels = cv.split(img) #np.mod(cmy, 255)
    channels_processed = []

    svg = vpype.VectorData()

    c = 0
    while c < len(channels):
        path = os.path.join(tmp_path, f"img-{c}.png")
        _, labels, centers = cv.kmeans(np.float32(channels[c]), KMEANS_ITERATIONS, None, KMEANS_CRITERIA, KMEANS_ATTEMPTS, KMEANS_FLAGS)
        #centers = (centers // 8) * 8
        centers = np.uint8(centers)
        cv.imwrite(path, channels[c])

        levels = []

        for l in np.unique(labels): # TODO: rewrite better
            x = []
            i = 0
            while i < len(labels):
                if labels[i] == l:
                    x.append(centers[l,i])
                i += 1
            avg = np.uint8(np.mean(x))
            levels.append(avg)

        #levels = np.sort(levels)[::-1] # sort in decending order

        #output_path = applySpiral(path, line_color=COLORS[c], levels=levels)
        #output_path = applyCrossed(path, line_color=COLORS[c], levels=levels)
        #output_path = applySketchy(path)
        output_path = os.path.join(tmp_path, f"img-{c}.svg")
        #output_path = tmp.getPath(f"img-{c}.svg")
        gen = Squiggle(path, output_path)
        #gen.output_path = output_path
        #gen.setImage(path)
        gen.generate(color=COLORS[c], x_offset=c, y_offset=c)
        print(output_path)

        #channel_processed_svg = hatched.hatch(path, hatch_pitch=4, levels=(20, 100, 180), blur_radius=1, image_scale=SCALE, show_plot=False)

        channel_processed = cv.imread(getOutputPngPath(output_path)) # , cv.IMREAD_GRAYSCALE
        channels_processed.append(channel_processed)

        #channel_processed_svg = vpype.read_svg(output_path, 0.1, simplify=False)
        #channel_processed_svg = svg2paths(output_path)
        #svg.add(channel_processed_svg, c)

        #paths, attributes, svg_attributes = svg2paths(output_path, return_svg_attributes=True)
        #wsvg(paths, attributes=attributes, svg_attributes=svg_attributes, filename=output_path, openinbrowser=False) # DEBUG: just for convenience

        c += 1

    #img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #_, img_gray = cv.threshold(img_gray, 63, 255,cv.THRESH_TRUNC)
    #gray_path = os.path.join(tmp_path, "img-bw.png")
    #cv.imwrite(gray_path, img_gray)
    #hatched.hatch(gray_path, hatch_pitch=4, levels=(20, 100, 180), blur_radius=0, image_scale=SCALE, show_plot=False)
    #output_path = applySpiral(gray_path, line_color="#000000", levels=levels)
    #output_path = applyCrossed(gray_path, levels=levels, line_color="#000000")

    #vpype.write_svg("test.svg", svg)
    #vpype.write_hpgl("test.hpgl", svg)

    #disp = cv.merge(channels_processed)
    #disp = CMY2BGR(disp)
    #cv.imwrite(os.path.join(tmp_path, "img-processed.png"), disp)
    #cv.imshow("image", np.hstack([disp]))
    #cv.imshow("image", np.hstack(channels_processed))
    while cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) == 1:
        if cv.waitKey(100) >= 0:
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()