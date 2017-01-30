#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import os
import random

import numpy as np
import pygame
from pygame import locals as pgl

from eyes import GridEye
from objects import ImageObj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", metavar="INPUT_DIR")
    args = parser.parse_args()

    path = args.input_dir
    assert os.path.isdir(path)
    names = os.listdir(path)
    name = random.choice(names)

    win_width, win_height, channels = 960, 960, 3
    pixel_size, sensor_size = 16, 1.1
    assert win_width % pixel_size == 0 and win_height % pixel_size == 0
    width, height = win_width // pixel_size, win_height // pixel_size
    step, angle = 1, 0.1

    eye = GridEye(pos=(0.5, 0.5, 100), up=(0, 1, 0), front=(0, 0, -1),
                  width=width, height=height, size=sensor_size)
    world = []
    obj = ImageObj(pos=(0, 0, 0), up=(0, 1, 0), front=(0, 0, -1), path=os.path.join(path, name))
    world.append(obj)

    pygame.init()
    canvas = pygame.display.set_mode((width * pixel_size, height * pixel_size))
    data = np.zeros((width, pixel_size, height, pixel_size, channels))
    view = data.reshape((width * pixel_size, height * pixel_size, channels))

    data[...] = eye.see(world)[:, None, :, None, :]
    move = None
    rot = None
    ctrl = False
    while True:
        for event in pygame.event.get():
            if event.type == pgl.KEYUP:
                move = None
                rot = None
                if event.key == pgl.K_LCTRL:
                    ctrl = False
            if event.type == pgl.KEYDOWN:
                if event.key == pgl.K_ESCAPE:
                    pygame.quit()
                    exit()
                elif event.key == pgl.K_LCTRL:
                    ctrl = True
                elif event.key == pgl.K_SPACE:
                    name = random.choice(names)
                    obj.set_image(os.path.join(path, name))
                elif event.key == pgl.K_LEFT:
                    move = (-step, 0, 0, True) if not ctrl else None
                    rot = ('u', angle) if ctrl else None
                elif event.key == pgl.K_RIGHT:
                    move = (step, 0, 0, True) if not ctrl else None
                    rot = ('u', -angle) if ctrl else None
                elif event.key == pgl.K_UP:
                    move = (0, step, 0, True) if not ctrl else None
                    rot = ('r', angle) if ctrl else None
                elif event.key == pgl.K_DOWN:
                    move = (0, -step, 0, True) if not ctrl else None
                    rot = ('r', -angle) if ctrl else None
                elif event.key == pgl.K_c:
                    move = None
                    rot = ('f', angle)
                elif event.key == pgl.K_x:
                    move = None
                    rot = ('f', -angle)
                elif event.key == pgl.K_a:
                    move = (0, 0, step, True)
                    rot = None
                elif event.key == pgl.K_z:
                    move = (0, 0, -step, True)
                    rot = None
                else:
                    move = None
                    rot = None

                    if event.key == pgl.K_b:
                        obj.mode = "bilinear"
                    elif event.key == pgl.K_n:
                        obj.mode = "nearest"

                    if event.key == pgl.K_p:
                        eye.fdist = None
                    elif event.key == pgl.K_f:
                        eye.fdist = 25
                    elif event.key == pgl.K_r:
                        eye.fdist = -25

        if move is not None:
            eye.move(*move)
        if rot is not None:
            eye.rot(*rot)
        data[:] = eye.see(world)[:, None, :, None, :]

        pygame.surfarray.blit_array(canvas, view)
        pygame.display.update()


if __name__ == "__main__":
    main()
