import os
import sys
import math
import xml.etree.ElementTree as ET
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, real, imag

from .config import *

def generate_gcode(paths):

    gcode = preamble

    # TODO: scale dynamically to bed size

    for path in paths:
        gcode += "\n" + path_to_gcode(path)

    gcode += postamble

    return gcode

def g_string(x, y, prefix="G1", p=4):
    return f"{prefix} X{SCALE*x:.{p}f} Y{-SCALE*y:.{p}f}\n" # DEBUG: y is negative as a quick fix

def path_to_gcode(path):
    #assert path.isclosed() # make sure path is closed

    x_start, y_start = real(path.start), imag(path.start)

    path_gcode = g_string(x_start, y_start, prefix="G0") + shape_preamble

    #path_gcode = f"""
    #G0 F{FEED_RATE} X{x_start} Y{y_start}
    #{shape_preamble}
    #"""

    for segment in path:
        x, y = real(segment.end), imag(segment.end)
        #path_gcode += f"G0 F{FEED_RATE} X{x_start} Y{y_start}"
        path_gcode += g_string(x, y)

    path_gcode += shape_postamble

    return path_gcode