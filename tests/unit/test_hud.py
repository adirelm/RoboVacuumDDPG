from src.gui import palette
from src.gui.hud import Hud
from src.services.live_session import Frame


def _frame():
    return Frame(
        mode="train",
        pose=(1.0, 1.0, 0.0),
        lidar_endpoints=[],
        new_cells=[],
        coverage=0.4,
        reward=1.5,
        collision=False,
        episode=3,
        step=42,
        sigma=0.12,
        buffer_size=128,
        done=False,
    )


def test_hud_draws_text(gui_surface):
    gui_surface.fill(palette.BG)
    Hud().draw(gui_surface, _frame(), "train", 4)
    found = any(
        gui_surface.get_at((x, y))[:3] != palette.BG for x in range(0, 420, 3) for y in range(0, 90, 3)
    )
    assert found


def test_hud_badge_branch_runs(gui_surface):
    Hud().draw(gui_surface, _frame(), "play", 1, no_checkpoint=True)  # exercises the badge path
