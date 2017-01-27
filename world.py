#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import os
import random

import numpy as np
import pygame
from pygame import locals as pgl
from scipy import misc


class Base(object):
    def __init__(self, pos, up, front):
        self._pos = np.array(pos, dtype="float32")
        self._up = np.array(up, dtype='float32')
        self._front = np.array(front, dtype='float32')
        self._right = np.cross(self._front, self._up)

    def move(self, dx, dy, dz):
        self._pos += np.array([dx, dy, dz])
        self._pos_updated()

    def _pos_updated(self):
        pass


class Eye(Base):
    def __init__(self, pos, up, front):
        super(Eye, self).__init__(pos, up, front)
        self._ray_pos = None
        self._ray_dir = None
        self._view = None

    def _clear(self):
        raise NotImplementedError

    def _update(self, view):
        raise NotImplementedError

    def _see(self, obj):
        return obj.view(self._ray_pos, self._ray_dir)

    def see(self, world):
        self._clear()
        for obj in world.objs:
            self._update(self._see(obj))
        return self._view


class GridEye(Eye):
    def __init__(self, pos, up, front, width, height, size):
        super(GridEye, self).__init__(pos, up, front)

        self._gx = np.linspace(0, 1, width)[:, None, None] * size * (width - 1)
        self._gy = np.linspace(1, 0, height)[None, :, None] * size * (height - 1)
        self._pos_updated()
        self._ray_dir = self._front[None, None, :]

        self._view = np.zeros((height, width, 3))

    def _clear(self):
        self._view[...] = 0

    def _update(self, view):
        self._view += view

    def _pos_updated(self):
        super(GridEye, self)._pos_updated()
        self._ray_pos = self._pos + self._right * self._gx + self._up * self._gy
        print(self._pos)


class Obj(Base):
    def __init__(self, pos, up, front):
        super(Obj, self).__init__(pos, up, front)

    def view(self, ray_pos, ray_dir):
        raise NotImplementedError


class FlatObj(Obj):
    def __init__(self, pos, up, front):
        super(FlatObj, self).__init__(pos, up, front)

    def _get(self, px, py):
        raise NotImplementedError

    def view(self, ray_pos, ray_dir):
        d = (self._pos - ray_pos).dot(self._front) / ray_dir.dot(self._front)
        intersections = d[:, :, None] * ray_dir + ray_pos
        print(intersections.shape)
        px = (intersections - self._pos).dot(self._right)
        py = (intersections - self._pos).dot(self._up)
        return self._get(px, py)


class ImageObj(FlatObj):
    def __init__(self, pos, up, front, path):
        super(ImageObj, self).__init__(pos, up, front)
        self._image = None
        self._w, self._h = None, None
        self.set_image(path)

    def _get(self, px, py):
        x1 = np.trunc(px).astype('int')
#        x2 = np.ceil(px).astype('int')
        y1 = np.trunc(py).astype('int')
#        y2 = np.ceil(py).astype('int')
        x1[(x1 < 0) | (x1 >= self._w)] = -1
#        x2[(x2 < 0) | (x2 >= self._w)] = -1
        y1[(y1 < 0) | (y1 >= self._h)] = -1
#        y2[(y2 < 0) | (y2 >= self._h)] = -1
#        ret = (self._image[x1, y1] + self._image[x1, y2] + self._image[x2, y1] + self._image[x2, y2]) / 4
        return self._image[x1, y1]

    def set_image(self, path):
        img = misc.imread(path, mode="RGB").transpose((1, 0, 2))[:, ::-1, :]
        self._w, self._h, c = img.shape
        self._image = np.zeros((self._w + 1, self._h + 1, c))
        self._image[:-1, :-1, :] = img


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
