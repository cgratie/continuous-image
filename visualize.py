#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import os
import random

import numpy as np
import pygame
from pygame import locals as pgl
from scipy import misc


class ImageViewer(object):
    def __init__(self, width, height, channels):
        self._left = 0
        self._top = 0
        self._width = width
        self._height = height
        self._step = 5
        self._image = None
        self._w, self._h = None, None
        self._view = np.zeros((width, height, channels))

    def read_image(self, path):
        self._image = misc.imread(path, mode="RGB").transpose((1, 0, 2))
        self._w, self._h, _ = self._image.shape
        self._update_view()

    def _update_view(self):
        self._view[...] = 0
        # copy into view [left:left+width, right:right+width]
        vx1, vx2, ix1, ix2 = 0, self._width, self._left, self._left + self._width
        vy1, vy2, iy1, iy2 = 0, self._height, self._top, self._top + self._height
        if ix1 < 0:
            vx1 += -ix1
            ix1 = 0
        if iy1 < 0:
            vy1 += -iy1
            iy1 = 0
        if ix2 > self._w:
            vx2 -= ix2 - self._w
            ix2 = self._w
        if iy2 > self._h:
            vy2 -= iy2 - self._h
            iy2 = self._h
        if vx1 < vx2 and vy1 < vy2:
            self._view[vx1:vx2, vy1:vy2, :] = self._image[ix1:ix2, iy1:iy2, :]

    def view(self):
        return self._view

    def move(self, dx, dy):
        self._left += dx
        self._top += dy
        self._update_view()


parser = argparse.ArgumentParser()
parser.add_argument("input_dir", metavar="INPUT_DIR")
args = parser.parse_args()

path = args.input_dir
assert os.path.isdir(path)
names = os.listdir(path)


width, height, channels, pixel_size = 120, 120, 3, 8
viewer = ImageViewer(width, height, channels)


def get_image():
    name = random.choice(names)
    viewer.read_image(os.path.join(path, name))
    return viewer.view()

image = get_image()


pygame.init()
canvas = pygame.display.set_mode((width * pixel_size, height * pixel_size))
data = np.zeros((width, pixel_size, height, pixel_size, channels))
view = data.reshape((width * pixel_size, height * pixel_size, channels))
data[...] = get_image()[:, None, :, None, :]
while True:
    for event in pygame.event.get():
        if event.type == pgl.KEYDOWN:
            if event.key == pgl.K_SPACE:
                image = get_image()
                data[:] = image[:, None, :, None, :]
            elif event.key == pgl.K_ESCAPE:
                pygame.quit()
                exit()
            elif event.key == pgl.K_LEFT:
                viewer.move(-10, 0)
                data[:] = viewer.view()[:, None, :, None, :]
            elif event.key == pgl.K_RIGHT:
                viewer.move(10, 0)
                data[:] = viewer.view()[:, None, :, None, :]
            elif event.key == pgl.K_UP:
                viewer.move(0, -10)
                data[:] = viewer.view()[:, None, :, None, :]
            elif event.key == pgl.K_DOWN:
                viewer.move(0, 10)
                data[:] = viewer.view()[:, None, :, None, :]

    pygame.surfarray.blit_array(canvas, view)
    pygame.display.update()
