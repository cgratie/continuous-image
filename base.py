from __future__ import print_function, division

import numpy as np


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
