import pygame

from src.gui.input_map import command_for, drive_action


def test_command_mapping_known_keys():
    assert command_for(pygame.K_SPACE) == "pause"
    assert command_for(pygame.K_t) == "mode_train"
    assert command_for(pygame.K_p) == "mode_play"
    assert command_for(pygame.K_d) == "mode_drive"
    assert command_for(pygame.K_ESCAPE) == "quit"
    assert command_for(pygame.K_l) == "toggle_lidar"
    assert command_for(pygame.K_TAB) == "cycle_map"


def test_command_mapping_unbound_key_is_none():
    assert command_for(pygame.K_5) is None


def test_drive_action_from_arrows():
    up_right = {pygame.K_UP: True, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: True}
    assert drive_action(up_right) == [1.0, 1.0]
    down_left = {pygame.K_UP: False, pygame.K_DOWN: True, pygame.K_LEFT: True, pygame.K_RIGHT: False}
    assert drive_action(down_left) == [-1.0, -1.0]
    none = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}
    assert drive_action(none) == [0.0, 0.0]
