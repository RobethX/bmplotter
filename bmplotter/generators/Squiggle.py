# TODO: say where this algorithm came from

from __future__ import division, print_function
import numpy as np
from PIL import Image
from svgpathtools import *
import svgpathtools
from svgpathtools.path import polyline
from svgwrite.mixins import ViewBox
from . import Generator

CLOSED_WARNING_ON=False # suppress svgpathtools warning

DEF_COLOR="#000000"

DEF_FREQUENCY = 192
MIN_FREQUENCY = 5
MAX_FREQUENCY = 256

DEF_LINE_COUNT = 70
MIN_LINE_COUNT = 10
MAX_LINE_COUNT = 200

DEF_AMPLITUDE = 2.0
MIN_AMPLITUDE = 0.1
MAX_AMPLITUDE = 5.0

DEF_SPACING = 1.5
MIN_SPACING = 0.5
MAX_SPACING = 2.9

DEF_HORIZONTAL_PHASE_SHIFT = 0
DEF_VERTICAL_PHASE_SHIFT = 0

MIN_POINTS = 5
BRIGHTNESS_THRESHOLD = 250

class Squiggle(Generator):
    def __init__(self, img=None, input_path=None, frequency=DEF_FREQUENCY, line_count=DEF_LINE_COUNT, amplitude=DEF_AMPLITUDE, spacing=DEF_SPACING): # TODO: use kwargs
        super().__init__(img, input_path)

        self.frequency = frequency
        self.line_count = line_count
        self.amplitude = amplitude
        self.spacing = spacing

    def setFrequency(self, frequency):
        self.frequency = np.clip(frequency, MIN_FREQUENCY, MAX_FREQUENCY)
        return self.frequency

    def setLineCount(self, line_count):
        self.line_count = np.clip(line_count, MIN_LINE_COUNT, MAX_LINE_COUNT)
        return self.line_count

    def setAmplitude(self, amplitude):
        self.amplitude = np.clip(amplitude, MIN_AMPLITUDE, MAX_AMPLITUDE)
        return self.amplitude

    def setSpacing(self, spacing):
        self.spacing = np.clip(spacing, MIN_SPACING, MAX_SPACING)
        return self.spacing

    def generate(self, color=DEF_COLOR, x_offset=DEF_HORIZONTAL_PHASE_SHIFT, y_offset=DEF_VERTICAL_PHASE_SHIFT, smooth=False, continuous=False):
        squiggles = []

        for y in range(0, self.img.height, self.img.height // self.line_count): # self.img.height // self.line_count
            y = (y + y_offset) % self.img.height
            a = 0
            current_line = [] # store bits of the line
            if continuous:
                current_line.append(complex(0, y)) # start the line

            for x in np.arange(self.spacing, self.img.width, self.spacing):
                v = np.mean(self.img.getpixel((x, y))) # TODO: downsample image and average chunk!!

                #r = (255 - v) / self.line_count * self.amplitude
                r = self.amplitude * (255 - v) / self.line_count
                a += (255 - v) / self.frequency

                point = complex(x, y + np.sin(a + x_offset)*r)

                if continuous:
                    current_line.append(point)
                else:
                    if (v < BRIGHTNESS_THRESHOLD or (len(current_line) > 0 and len(current_line) < MIN_POINTS)): # TODO: calculate this with greyscale mask
                        current_line.append(point)
                    else:
                        squiggles.extend(polyline(*current_line))
                        current_line = []

            current_line_path = polyline(*current_line)
            if smooth and continuous: # TODO: fix smoothing on discontinuous paths
                current_line_path = smoothed_path(current_line_path)
            squiggles.extend(current_line_path)

        paths = Path(*squiggles).continuous_subpaths()
        #wsvg(paths, filename=self.output_path, colors=([color]*len(paths)))

        #return self.output_path
        return paths

if __name__ == "__main__":
    gen = Squiggle()