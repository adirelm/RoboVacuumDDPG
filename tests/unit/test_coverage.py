import math

from src.env.coverage import CoverageGrid

BOUNDS = (0.0, 0.0, 4.0, 4.0)
CELL = 0.10
CLEAN_R = 0.17


def test_first_mark_cleans_some_then_remark_cleans_none():
    grid = CoverageGrid(BOUNDS, CELL, CLEAN_R)
    first = grid.mark(2.0, 2.0)
    assert first > 0  # disc of radius 0.17 covers >=1 cell centre
    again = grid.mark(2.0, 2.0)  # C-2: no reward for revisiting
    assert again == 0


def test_fraction_rises_then_resets():
    grid = CoverageGrid(BOUNDS, CELL, CLEAN_R)
    assert grid.fraction() == 0.0
    grid.mark(2.0, 2.0)
    f = grid.fraction()
    assert 0.0 < f < 1.0  # a single disc never covers a 4x4 room
    grid.reset()  # C-3: cleared on reset
    assert grid.fraction() == 0.0


def test_nearest_uncleaned_bearing_is_unit_vector():
    grid = CoverageGrid(BOUNDS, CELL, CLEAN_R)
    grid.mark(2.0, 2.0)  # clean a patch near the centre
    cos_b, sin_b = grid.nearest_uncleaned_bearing(2.0, 2.0, 0.0)
    assert math.isclose(math.hypot(cos_b, sin_b), 1.0, abs_tol=1e-6)


def test_bearing_zero_when_fully_cleaned():
    # tiny room: one cell, clean it, then bearing is (0,0)
    grid = CoverageGrid((0.0, 0.0, 0.05, 0.05), CELL, CLEAN_R)
    grid.mark(0.0, 0.0)
    assert grid.fraction() == 1.0
    assert grid.nearest_uncleaned_bearing(0.0, 0.0, 0.0) == (0.0, 0.0)


def test_fraction_denominator_is_reachable_free_space_not_whole_bbox():
    # bbox is 0..4 in x but only the left half (x < 2) is inside the floor plan;
    # the right half are wall/void cells that can never be cleaned.
    def is_free(x: float, y: float) -> bool:
        return x < 2.0

    grid = CoverageGrid(BOUNDS, CELL, CLEAN_R, is_free=is_free)
    # sweep the whole FREE half on a fine lattice; the void half stays uncleaned.
    ys = [c * CELL for c in range(41)]
    xs = [c * CELL for c in range(20)]  # x in [0, 1.9] -> all inside free space
    for x in xs:
        for y in ys:
            grid.mark(x, y)
    # all reachable free cells cleaned => fraction must hit 1.0 despite the
    # uncleanable right half still sitting in the bounding box.
    assert grid.fraction() == 1.0


def test_fraction_default_is_whole_bbox_backward_compatible():
    # No is_free -> every bbox cell counts (legacy convex-box behaviour).
    grid = CoverageGrid((0.0, 0.0, 0.05, 0.05), CELL, CLEAN_R)
    grid.mark(0.0, 0.0)
    assert grid.fraction() == 1.0


def test_bearing_is_in_robot_frame():
    # nearest uncleaned cell straight ahead in world (+x); robot heading +x ->
    # robot-frame bearing ~ (1, 0). Heading rotated by pi/2 -> bearing ~ (0, -1).
    grid = CoverageGrid((0.0, 0.0, 1.0, 1.0), CELL, CLEAN_R)
    grid.mark(0.0, 0.5)  # clean near the left edge, leave +x side uncleaned
    cf, _ = grid.nearest_uncleaned_bearing(0.0, 0.5, 0.0)
    _cf_rot, sf_rot = grid.nearest_uncleaned_bearing(0.0, 0.5, math.pi / 2)
    assert cf > 0.0  # uncleaned mass is ahead in the robot frame
    assert sf_rot < cf  # rotating the heading rotates the robot-frame bearing
