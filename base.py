from __future__ import print_function, division

import math

import numpy as np


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
            self._up = self._rot(self._up, self._front, angle)
            self._right = self._rot(self._right, self._front, angle)
        elif axis == 'r':
            self._up = self._rot(self._up, self._right, angle)
            self._front = self._rot(self._front, self._right, angle)
        elif axis == 'u':
            self._front = self._rot(self._front, self._up, angle)
            self._right = self._rot(self._right, self._up, angle)
        self._dir_updated()

    @staticmethod
    def _rot(p, r, t):
        x, y, z = p
        u, v, w = r
        dot = u * x + v * y + w * z
        ct = math.cos(t)
        st = math.sin(t)
        xx = u * dot * (1 - ct) + x * ct + (- w * y + v * z) * st
        yy = v * dot * (1 - ct) + y * ct + (w * x - u * z) * st
        zz = w * dot * (1 - ct) + z * ct + (- v * x + w * y) * st
        return np.array((xx, yy, zz))

    def _pos_updated(self):
        pass

    def _dir_updated(self):
        pass
