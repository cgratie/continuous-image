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
