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

    width, height, channels, pixel_size, sensor_size, step = 120, 120, 3, 8, 1, 1
    eye = GridEye(pos=(0.5, 0.5, 1), up=(0, 1, 0), front=(0, 0, -1), width=width, height=height, size=sensor_size)
    world = []
    obj = ImageObj(pos=(0, 0, 0), up=(0, 1, 0), front=(0, 0, -1), path=os.path.join(path, name))
    world.append(obj)

    pygame.init()
    canvas = pygame.display.set_mode((width * pixel_size, height * pixel_size))
    data = np.zeros((width, pixel_size, height, pixel_size, channels))
    view = data.reshape((width * pixel_size, height * pixel_size, channels))

    data[...] = eye.see(world)[:, None, :, None, :]
    move = None
    while True:
        for event in pygame.event.get():
            if event.type == pgl.KEYUP:
                move = None
            if event.type == pgl.KEYDOWN:
                if event.key == pgl.K_ESCAPE:
                    pygame.quit()
                    exit()
                elif event.key == pgl.K_SPACE:
                    name = random.choice(names)
                    obj.set_image(os.path.join(path, name))
                elif event.key == pgl.K_LEFT:
                    move = (-step, 0, 0)
                elif event.key == pgl.K_RIGHT:
                    move = (step, 0, 0)
                elif event.key == pgl.K_UP:
                    move = (0, step, 0)
                elif event.key == pgl.K_DOWN:
                    move = (0, -step, 0)
                else:
                    move = None

        if move is not None:
            eye.move(*move)
        data[:] = eye.see(world)[:, None, :, None, :]

        pygame.surfarray.blit_array(canvas, view)
        pygame.display.update()


if __name__ == "__main__":
    main()
