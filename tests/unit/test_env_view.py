from src.gui import palette
from src.gui.env_view import EnvScene, draw_env
from src.gui.transform import View

WALLS = [(0.0, 0.0, 4.0, 0.0), (4.0, 0.0, 4.0, 4.0), (4.0, 4.0, 0.0, 4.0), (0.0, 4.0, 0.0, 0.0)]


def _scene(**kw):
    base = {
        "walls": WALLS,
        "pose": (2.0, 2.0, 0.0),
        "robot_radius": 0.17,
        "cell_size": 0.5,
        "covered": [(2.0, 2.0)],
        "trail": [(1.0, 1.0), (2.0, 2.0)],
        "lidar_endpoints": [(3.9, 2.0)],
    }
    base.update(kw)
    return EnvScene(**base)


def test_draw_env_runs_and_paints_robot_disc(gui_surface):
    view = View((0.0, 0.0, 4.0, 4.0), 900, 620)
    draw_env(gui_surface, view, _scene())
    cx, cy = view.to_screen(2.0, 2.0)
    # heading points +x, so sample a pixel below centre — on the disc, off the heading line
    assert gui_surface.get_at((cx, cy + 8))[:3] == palette.ROBOT


def test_corner_is_background(gui_surface):
    view = View((0.0, 0.0, 4.0, 4.0), 900, 620)
    draw_env(gui_surface, view, _scene())
    assert gui_surface.get_at((0, 0))[:3] == palette.BG


def test_coverage_toggle_hides_cells(gui_surface):
    view = View((0.0, 0.0, 4.0, 4.0), 900, 620)
    # a covered cell away from the robot so the robot disc doesn't overpaint it
    draw_env(gui_surface, view, _scene(covered=[(0.5, 0.5)], show_coverage=True))
    px, py = view.to_screen(0.5, 0.5)
    assert gui_surface.get_at((px, py))[:3] == palette.COVERED

    draw_env(gui_surface, view, _scene(covered=[(0.5, 0.5)], show_coverage=False))
    assert gui_surface.get_at((px, py))[:3] == palette.BG
