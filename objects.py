from __future__ import print_function, division

import numpy as np
from scipy import misc

from base import Base


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
        px = (intersections - self._pos).dot(self._right)
        px[d < 0] = np.inf
        py = (intersections - self._pos).dot(self._up)
        py[d < 0] = np.inf
        return self._get(px, py)


class ImageObj(FlatObj):
    def __init__(self, pos, up, front, path):
        super(ImageObj, self).__init__(pos, up, front)
        self._image = None
        self._w, self._h = None, None
        self.set_image(path)

    def _get(self, px, py):
        x = np.trunc(px + self._w // 2).astype('int')
        y = np.trunc(py + self._h // 2).astype('int')
        x[(x < 0) | (x >= self._w)] = -1
        y[(y < 0) | (y >= self._h)] = -1
        return self._image[x, y]

    def set_image(self, path):
        img = misc.imread(path, mode="RGB").transpose((1, 0, 2))[:, ::-1, :]
        self._w, self._h, c = img.shape
        self._image = np.zeros((self._w + 1, self._h + 1, c))
        self._image[:-1, :-1, :] = img
