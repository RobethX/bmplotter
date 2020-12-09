import os
import argparse
import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt

from svgpathtools import svg2paths, wsvg

import utils
import generators
import svg2gcode

NUM_SHADES = 4 # number of shading levels
#COLORS = ["#FF0000", "#00FF00", "#0000FF"] # TODO: svgpathtools has a utility for converting hex colors
#COLORS = ["#00FFFF", "#FF00FF", "#FFFF00", "#000000"] # CMYK
COLORS = ["#FFFF00", "#FF00FF", "#00FFFF"] # YMC
#COLORS = ["#0000FF", "#00FF00", "#FF0000"] # BGR

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

    #img = utils.image.removeWhite(img)

    tmp = utils.path.TempFolder()

    tmp_path = os.path.normpath("tmp")
    #tmp_path = tmp.getPath()
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    #cmyk = utils.color.BGR2CMYK(img)
    #cmy = utils.color.BGR2CMY(img)

    channels = cv.split(img) #np.mod(cmy, 255)
    channels_processed = []

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

        output_path = os.path.join(tmp_path, f"img-{c}")
        #output_path = tmp.getPath(f"img-{c}")
        gen = generators.Squiggle(path, f"{output_path}.svg")
        #gen.output_path = output_path
        #gen.setImage(path)
        paths = gen.generate(color=COLORS[c], x_offset=c, y_offset=c)
        print(output_path)


        paths_optimized = utils.svg.optimize(paths)
        wsvg(paths_optimized, filename=f"{output_path}-optimized.svg", colors=([COLORS[c]]*len(paths)))

        gcode = svg2gcode.generate_gcode(paths_optimized)
        gcode_file = open(f"{output_path}.gcode", "w")
        gcode_file.write(gcode)
        gcode_file.close

        channel_processed = cv.imread(getOutputPngPath(output_path)) # , cv.IMREAD_GRAYSCALE
        channels_processed.append(channel_processed)

        #wsvg(paths, attributes=attributes, svg_attributes=svg_attributes, filename=output_path, openinbrowser=False) # DEBUG: just for convenience

        c += 1

    #img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #_, img_gray = cv.threshold(img_gray, 63, 255,cv.THRESH_TRUNC)
    #gray_path = os.path.join(tmp_path, "img-bw.png")
    #cv.imwrite(gray_path, img_gray)
    #output_path = applySpiral(gray_path, line_color="#000000", levels=levels)
    #output_path = applyCrossed(gray_path, levels=levels, line_color="#000000")

    #disp = cv.merge(channels_processed)
    #disp = utils.color.BGR(disp)
    #cv.imwrite(os.path.join(tmp_path, "img-processed.png"), disp)
    #cv.imshow("image", np.hstack([disp]))
    #cv.imshow("image", np.hstack(channels_processed))
    #while cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) == 1:
    #    if cv.waitKey(100) >= 0:
    #        break
    #cv.destroyAllWindows()

if __name__ == "__main__":
    main()