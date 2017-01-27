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


class World(object):
    def __init__(self):
        self.objs = []

    def add_obj(self, obj):
        self.objs.append(obj)


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
    world = World()
    obj = ImageObj(pos=(0, 0, 0), up=(0, 1, 0), front=(0, 0, -1), path=os.path.join(path, name))
    world.add_obj(obj)

    pygame.init()
    canvas = pygame.display.set_mode((width * pixel_size, height * pixel_size))
    data = np.zeros((width, pixel_size, height, pixel_size, channels))
    view = data.reshape((width * pixel_size, height * pixel_size, channels))

    data[...] = eye.see(world)[:, None, :, None, :]
    while True:
        for event in pygame.event.get():
            if event.type == pgl.KEYDOWN:
                if event.key == pgl.K_SPACE:
                    name = random.choice(names)
                    obj.set_image(os.path.join(path, name))
                    data[:] = eye.see(world)[:, None, :, None, :]
                elif event.key == pgl.K_ESCAPE:
                    pygame.quit()
                    exit()
                elif event.key == pgl.K_LEFT:
                    eye.move(-step, 0, 0)
                    data[:] = eye.see(world)[:, None, :, None, :]
                elif event.key == pgl.K_RIGHT:
                    eye.move(step, 0, 0)
                    data[:] = eye.see(world)[:, None, :, None, :]
                elif event.key == pgl.K_UP:
                    eye.move(0, step, 0)
                    data[:] = eye.see(world)[:, None, :, None, :]
                elif event.key == pgl.K_DOWN:
                    eye.move(0, -step, 0)
                    data[:] = eye.see(world)[:, None, :, None, :]

        pygame.surfarray.blit_array(canvas, view)
        pygame.display.update()


if __name__ == "__main__":
    main()
