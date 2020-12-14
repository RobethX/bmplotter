import os
import argparse
import logging
import cv2 as cv
import numpy as np

from svgpathtools import svg2paths, wsvg

import utils
import generators
import svg2gcode

NUM_SHADES = 4 # number of shading levels
#COLORS = ["#FF0000", "#00FF00", "#0000FF"] # TODO: svgpathtools has a utility for converting hex colors
#COLORS = ["#00FFFF", "#FF00FF", "#FFFF00", "#000000"] # CMYK
#COLORS = ["#FFFF00", "#FF00FF", "#00FFFF"] # YMC
#COLORS = ["#0000FF", "#00FF00", "#FF0000"] # BGR
COLORS = ["#FFFF00", "#FF00FF", "#00FFFF", "#000000"] # YMCK

# K-Means parameters
KMEANS_ACCURACY = 0.85 # percent
KMEANS_ITERATIONS = NUM_SHADES
KMEANS_ATTEMPTS = 100
KMEANS_FLAGS = cv.KMEANS_RANDOM_CENTERS
KMEANS_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# DEFAULT PARAMETERS
DEF_NUM_COLORS = 8

def processImage():
    img_raw = cv.imread(args.filename) # load image
    img = img_raw

    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #img = cv.medianBlur(img,5)
    #img = cv.filter2D(img, -1, kernel)
    img = cv.bilateralFilter(img, 9, 50, 50) # bilateral filter
    #img = cv.ximgproc.anisotropicDiffusion(img, 0.5, 0.02, 10) # 2d anisotropic diffusion

    # DEBUG: show image
    cv.imshow("image", np.hstack([img_raw, img]))
    while cv.getWindowProperty("image", cv.WND_PROP_VISIBLE) == 1:
        if cv.waitKey(100) >= 0:
            break
    cv.destroyAllWindows()

    return img

def main():
    parser = argparse.ArgumentParser(prog="painter-cam")
    parser.add_argument("filename", help="input image to convert")
    parser.add_argument("-c", help="amount of paint colors", action="store", dest="colors", default=DEF_NUM_COLORS, type=int)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true", dest="verbose")
    parser.add_argument("--log-file", help="log file path", action="store", dest="log_file", default=None, type=str)
    parser.add_argument("--no-black", help="skip black layer", action="store_false", dest="black_layer")

    global args
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, format="%(levelname)s: %(message)s")
    log = logging.getLogger()
    if args.verbose:
        #log.setLevel(logging.DEBUG)
        log.setLevel(logging.INFO)

    # TODO: check if file exists
    #img = processImage()
    img = cv.imread(args.filename)

    tmp = utils.path.TempFolder()

    tmp_path = os.path.normpath("tmp")
    #tmp_path = tmp.getPath()
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    channels = cv.split(img) #np.mod(cmy, 255)
    if args.black_layer:
        channels.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

    c = 0
    while c < len(channels):
        log.info(f"Vectorizing layer {c}...")
        path = os.path.join(tmp_path, f"img-{c}.png")
        cv.imwrite(path, channels[c])
        output_path = os.path.join(tmp_path, f"img-{c}")
        #output_path = tmp.getPath(f"img-{c}")
        gen = generators.Squiggle(img=channels[c]) #f"{output_path}.svg"
        paths = gen.generate(color=COLORS[c], x_offset=c, y_offset=c)

        log.info(f"Optimizing paths for layer {c}...")
        paths_optimized = utils.svg.optimize(paths)
        wsvg(paths_optimized, filename=f"{output_path}.svg", colors=([COLORS[c]]*len(paths)))

        log.info(f"Generating g-code for layer {c}...")
        gcode = svg2gcode.generate_gcode(paths_optimized)
        gcode_file = open(f"{output_path}.gcode", "w")
        gcode_file.write(gcode)
        gcode_file.close
        log.info(f"Saving g-code to {output_path}.gcode")

        #wsvg(paths, attributes=attributes, svg_attributes=svg_attributes, filename=output_path, openinbrowser=False) # DEBUG: just for convenience

        c += 1

if __name__ == "__main__":
    main()