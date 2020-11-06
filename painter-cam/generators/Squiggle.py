from __future__ import division, print_function
import numpy as np
from PIL import Image
from svgpathtools import *
from svgpathtools.path import polyline
from svgwrite.mixins import ViewBox
from . import Generator

CLOSED_WARNING_ON=False # suppress svgpathtools warning

DEF_FREQUENCY = 128
MIN_FREQUENCY = 5
MAX_FREQUENCY = 256

DEF_LINE_COUNT = 100
MIN_LINE_COUNT = 10
MAX_LINE_COUNT = 200

DEF_AMPLITUDE = 2.0
MIN_AMPLITUDE = 0.1
MAX_AMPLITUDE = 5.0

DEF_SPACING = 0.8
MIN_SPACING = 0.5
MAX_SPACING = 2.9

class Squiggle(Generator):
    def __init__(self, image_path):
        super().__init__(image_path)

        self.frequency = DEF_FREQUENCY
        self.lineCount = DEF_LINE_COUNT
        self.amplitude = DEF_AMPLITUDE
        self.spacing = DEF_SPACING

        squiggles = Path(*self.generate()).continuous_subpaths()

        disvg(squiggles)
        print(squiggles.count)
        #disvg(Path(*squiggles))

    def setFrequency(self, frequency):
        self.frequency = np.clip(frequency, MIN_FREQUENCY, MAX_FREQUENCY)
        return self.frequency

    def setLineCount(self, lineCount):
        self.lineCount = np.clip(lineCount, MIN_LINE_COUNT, MAX_LINE_COUNT)
        return self.lineCount

    def setAmplitude(self, amplitude):
        self.amplitude = np.clip(amplitude, MIN_AMPLITUDE, MAX_AMPLITUDE)
        return self.amplitude

    def setSpacing(self, spacing):
        self.spacing = np.clip(spacing, MIN_SPACING, MAX_SPACING)
        return self.spacing

    def generate(self):
        squiggleData = []
        squiggles = []
        #squiggles = polyline()
        #squiggles = Document()

        for y in range(0, self.img.height, self.img.height // self.lineCount): # self.img.height // self.lineCount
            a = 0
            currentLine = [] # store bits of the line
            currentLine.append(complex(0, y)) # start the line

            for x in np.arange(self.spacing, self.img.width, self.spacing):
                v = np.mean(self.img.getpixel((x, y)))

                #r = (255 - v) / self.lineCount * self.amplitude
                r = self.amplitude * (255 - v) / self.lineCount
                a += (255 - v) / self.frequency

                currentLine.append(complex(x, y + np.sin(a)*r))

            #squiggleData.append(currentLine)
            squiggles.extend(polyline(*currentLine))
            #squiggles.add_path(polyline(*currentLine))

        return squiggles

if __name__ == "__main__":
    gen = Squiggle()