from src.gui.transform import View


def test_corners_map_inside_window():
    v = View((0.0, 0.0, 4.0, 4.0), 900, 620, pad=20)
    for x, y in [(0, 0), (4, 0), (0, 4), (4, 4)]:
        px, py = v.to_screen(x, y)
        assert 0 <= px <= 900 and 0 <= py <= 620


def test_y_is_flipped():
    v = View((0.0, 0.0, 4.0, 4.0), 900, 620)
    top = v.to_screen(0.0, 4.0)
    bot = v.to_screen(0.0, 0.0)
    assert top[1] < bot[1]  # higher world-y -> smaller screen-y


def test_aspect_preserved_square_metre():
    # 1 m along x and 1 m along y must scale to the same pixel length.
    v = View((0.0, 0.0, 4.0, 2.0), 900, 620)
    p0 = v.to_screen(0.0, 0.0)
    px1 = v.to_screen(1.0, 0.0)
    py1 = v.to_screen(0.0, 1.0)
    dx = px1[0] - p0[0]
    dy = p0[1] - py1[1]
    assert abs(dx - dy) <= 1


def test_square_map_centre_maps_to_window_centre():
    v = View((0.0, 0.0, 4.0, 4.0), 900, 620, pad=20)
    px, py = v.to_screen(2.0, 2.0)
    assert abs(px - 450) <= 1 and abs(py - 310) <= 1


def test_scale_len_positive():
    v = View((0.0, 0.0, 4.0, 4.0), 900, 620)
    assert v.scale_len(1.0) > 0
    assert v.scale_len(2.0) == 2 * v.scale_len(1.0)
