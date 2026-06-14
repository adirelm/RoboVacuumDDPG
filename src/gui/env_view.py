"""Render the 2D vacuum world onto a pygame Surface: coverage, lidar, trail, walls, robot."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

from src.gui import palette
from src.gui.transform import View

_MIN_TRAIL_PTS = 2  # need at least two points to draw a polyline


@dataclass
class EnvScene:
    """Everything needed to draw one frame of the world (assembled by the App)."""

    walls: list[tuple[float, float, float, float]]
    pose: tuple[float, float, float]
    robot_radius: float
    cell_size: float
    covered: list[tuple[float, float]] = field(default_factory=list)
    trail: list[tuple[float, float]] = field(default_factory=list)
    lidar_endpoints: list[tuple[float, float]] = field(default_factory=list)
    show_lidar: bool = True
    show_coverage: bool = True


def draw_env(surface: pygame.Surface, view: View, scene: EnvScene) -> None:
    """Draw the full scene for one frame onto `surface` via the world->screen `view`."""
    surface.fill(palette.BG)
    if scene.show_coverage:
        _draw_coverage(surface, view, scene.covered, scene.cell_size)
    if scene.show_lidar:
        _draw_lidar(surface, view, scene.pose, scene.lidar_endpoints)
    _draw_trail(surface, view, scene.trail)
    _draw_walls(surface, view, scene.walls)
    _draw_robot(surface, view, scene.pose, scene.robot_radius)


def _draw_coverage(surface, view, covered, cell_size):
    side = max(view.scale_len(cell_size), 2)
    half = cell_size / 2.0
    for cx, cy in covered:
        left, top = view.to_screen(cx - half, cy + half)  # +half in world-y = top on screen
        surface.fill(palette.COVERED, pygame.Rect(left, top, side, side))


def _draw_lidar(surface, view, pose, endpoints):
    origin = view.to_screen(pose[0], pose[1])
    for ex, ey in endpoints:
        pygame.draw.line(surface, palette.LIDAR, origin, view.to_screen(ex, ey), 1)


def _draw_trail(surface, view, trail):
    if len(trail) >= _MIN_TRAIL_PTS:
        pts = [view.to_screen(x, y) for x, y in trail]
        pygame.draw.lines(surface, palette.TRAIL, False, pts, palette.TRAIL_W)


def _draw_walls(surface, view, walls):
    for x1, y1, x2, y2 in walls:
        a, b = view.to_screen(x1, y1), view.to_screen(x2, y2)
        pygame.draw.line(surface, palette.WALL, a, b, palette.WALL_W)


def _draw_robot(surface, view, pose, robot_radius):
    x, y, theta = pose
    centre = view.to_screen(x, y)
    pygame.draw.circle(surface, palette.ROBOT, centre, max(view.scale_len(robot_radius), 3))
    nose = view.to_screen(x + 1.6 * robot_radius * math.cos(theta), y + 1.6 * robot_radius * math.sin(theta))
    pygame.draw.line(surface, palette.HEADING, centre, nose, 2)
