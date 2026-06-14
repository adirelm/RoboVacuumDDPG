"""GUI colour palette + line widths.

Local visual-styling constants (CLAUDE.md §4 explicitly keeps colours/line widths
in their rendering module rather than config — they are visual design, not tunable
algorithm parameters).
"""

BG = (24, 26, 30)  # window background
WALL = (12, 12, 14)  # floor-plan boundary
ROBOT = (66, 135, 245)  # robot body disc
HEADING = (240, 240, 245)  # heading line
TRAIL = (90, 160, 250)  # path trail
COVERED = (46, 160, 90)  # cleaned cells
LIDAR = (210, 130, 60)  # lidar rays
TEXT = (232, 232, 236)  # HUD text
BADGE = (220, 70, 70)  # "no checkpoint" badge
CURVE = (90, 200, 130)  # reward curve
ROLL = (240, 200, 90)  # rolling-mean line
ZERO_LINE = (95, 95, 100)  # curve zero axis

WALL_W = 3
TRAIL_W = 2
CURVE_W = 2
