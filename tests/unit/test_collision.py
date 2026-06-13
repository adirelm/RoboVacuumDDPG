from src.env.collision import collides

ROBOT_R = 0.17


def test_collision_when_wall_within_radius():
    walls = [(0.10, -1.0, 0.10, 1.0)]  # distance 0.10 < 0.17
    assert collides(0.0, 0.0, ROBOT_R, walls) is True


def test_no_collision_when_wall_outside_radius():
    walls = [(0.20, -1.0, 0.20, 1.0)]  # distance 0.20 > 0.17
    assert collides(0.0, 0.0, ROBOT_R, walls) is False


def test_endpoint_clamped_distance():
    # nearest point on the segment is its endpoint (0.0, 0.30); distance 0.30 > 0.17
    walls = [(0.0, 0.30, 1.0, 0.30)]
    assert collides(0.0, 0.0, ROBOT_R, walls) is False
    # move the wall endpoint closer: nearest point endpoint (0.0, 0.10) -> 0.10 < 0.17
    walls = [(0.0, 0.10, 1.0, 0.40)]
    assert collides(0.0, 0.0, ROBOT_R, walls) is True


def test_no_walls_means_no_collision():
    assert collides(0.0, 0.0, ROBOT_R, []) is False


def test_collision_against_nearest_of_many():
    walls = [(5.0, -1.0, 5.0, 1.0), (0.05, -1.0, 0.05, 1.0)]
    assert collides(0.0, 0.0, ROBOT_R, walls) is True


def test_degenerate_zero_length_segment_uses_point_distance():
    # A zero-length "wall" (a point) exercises the denom == 0 guard: distance is
    # the plain point-to-point distance, not a divide-by-zero.
    assert collides(0.0, 0.0, ROBOT_R, [(1.0, 1.0, 1.0, 1.0)]) is False  # point ~1.41 away
    assert collides(0.0, 0.0, ROBOT_R, [(0.10, 0.0, 0.10, 0.0)]) is True  # point 0.10 < 0.17
