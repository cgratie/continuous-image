from __future__ import print_function, division

import numpy as np

from base import Base


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

    def see(self, objs):
        self._clear()
        for obj in objs:
            self._update(self._see(obj))
        return self._view


class GridEye(Eye):
    def __init__(self, pos, up, front, width, height, size, fdist=None):
        super(GridEye, self).__init__(pos, up, front)

        self._gx = np.linspace(-0.5, 0.5, width)[:, None, None] * size * (width - 1)  # type: np.ndarray
        self._gy = np.linspace(0.5, -0.5, height)[None, :, None] * size * (height - 1)  # type: np.ndarray
        self._fdist = fdist
        self._update_ray()

        self._view = np.zeros((height, width, 3))

    @property
    def fdist(self):
        return self._fdist

    @fdist.setter
    def fdist(self, val):
        self._fdist = val
        self._update_ray()

    def _clear(self):
        self._view[...] = 0

    def _update(self, view):
        self._view += view

    def _pos_updated(self):
        super(GridEye, self)._pos_updated()
        self._update_ray()

    def _dir_updated(self):
        super(GridEye, self)._dir_updated()
        self._update_ray()

    def _update_ray(self):
        if self._fdist is None:
            self._ray_pos = self._pos + self._right * self._gx + self._up * self._gy
            self._ray_dir = self._front[None, None, :]
        else:
            self._ray_pos = (self._pos + self._fdist * self._front)[None, None, :]
            if self._fdist > 0:
                self._ray_dir = self._fdist * self._front - (self._right * self._gx + self._up * self._gy)
            else:
                self._ray_dir = (self._right * self._gx + self._up * self._gy) - self._fdist * self._front
