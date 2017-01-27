from __future__ import print_function, division

import numpy as np

from util import rotation


class Base(object):
    def __init__(self, pos, up, front):
        self._pos = np.array(pos, dtype="float32")
        self._up = np.array(up, dtype='float32')
        self._front = np.array(front, dtype='float32')
        self._right = np.cross(self._front, self._up)

    def move(self, dx, dy, dz, relative=False):
        if relative:
            self._pos += dx * self._right + dy * self._up + dz * self._front
        else:
            self._pos += np.array([dx, dy, dz])
        self._pos_updated()

    def rot(self, axis, angle):
        if axis == 'f':
            self._up = rotation(self._up, self._front, angle)
            self._right = rotation(self._right, self._front, angle)
        elif axis == 'r':
            self._up = rotation(self._up, self._right, angle)
            self._front = rotation(self._front, self._right, angle)
        elif axis == 'u':
            self._front = rotation(self._front, self._up, angle)
            self._right = rotation(self._right, self._up, angle)
        self._dir_updated()

    def _pos_updated(self):
        pass

    def _dir_updated(self):
        pass
