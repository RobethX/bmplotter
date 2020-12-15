"""Constants and G-code parameters"""
FEED_RATE = 4500 # mm/min
DELAY = 0.100 # wait time between shapes
SCALE = 0.125

"""G-code commands"""
CMD_PEN_UP = "M5 S0" # raise pen command
CMD_PEN_DOWN = "M3 S1000" # lower pen command
CMD_SET_FEED_RATE = f"G1 F{FEED_RATE}" # set feed rate command
CMD_DWELL = f"G4 P{DELAY}" # pause for DELAY ms

"""G-code emitted at the start of processing the SVG file"""
preamble = f"""{CMD_PEN_UP} ; raise pen
G90 ; absolute positioning
G21 ; metric
{CMD_SET_FEED_RATE} ; set feed rate
"""

"""G-code emitted at the end of processing the SVG file"""
postamble = f"""
{CMD_PEN_UP} ; make sure pen is raised
G0 X0 Y0 ; return to zero
"""

"""G-code emitted before processing a SVG shape"""
shape_preamble = f"{CMD_PEN_DOWN} ; lower pen\n"

"""G-code emitted after processing a SVG shape"""
shape_postamble = f"{CMD_PEN_UP} ; raise pen\n{CMD_DWELL} ; delay\n"