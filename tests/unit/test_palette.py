from src.gui import palette

_COLOURS = [
    "BG",
    "WALL",
    "ROBOT",
    "HEADING",
    "TRAIL",
    "COVERED",
    "LIDAR",
    "TEXT",
    "BADGE",
    "CURVE",
    "ROLL",
    "ZERO_LINE",
]


def test_all_colours_are_rgb_int_triples():
    for name in _COLOURS:
        c = getattr(palette, name)
        assert isinstance(c, tuple) and len(c) == 3, f"{name} not a 3-tuple"
        assert all(isinstance(v, int) and 0 <= v <= 255 for v in c), f"{name} out of 0..255"


def test_line_widths_are_positive_ints():
    for name in ["WALL_W", "TRAIL_W", "CURVE_W"]:
        w = getattr(palette, name)
        assert isinstance(w, int) and w > 0
