import math

import numpy as np


def rot_rodrigues(point, axis, angle):
    x, y, z = point
    u, v, w = axis
    ct = math.cos(angle)
    st = math.sin(angle)
    dot = (u * x + v * y + w * z) * (1 - ct)
    xx = u * dot + x * ct + (- w * y + v * z) * st
    yy = v * dot + y * ct + (w * x - u * z) * st
    zz = w * dot + z * ct + (- v * x + u * y) * st
    return np.array((xx, yy, zz))


def rot_rodrigues_np(point, axis, angle):
    v = point
    k = axis
    st = np.sin(angle)
    ct = np.cos(angle)
    return v * ct + np.cross(k, v) * st + k * np.dot(k, v) * (1 - ct)


def rot_euler_rodrigues(point, axis, angle):
    x1, x2, x3 = point
    a = math.cos(angle / 2)
    k1, k2, k3 = axis
    st2 = math.sin(angle / 2)
    w1 = k1 * st2
    w2 = k2 * st2
    w3 = k3 * st2
    wx1 = w2 * x3 - w3 * x2
    wx2 = w3 * x1 - w1 * x3
    wx3 = w1 * x2 - w2 * x1
    wwx1 = w2 * wx3 - w3 * wx2
    wwx2 = w3 * wx1 - w1 * wx3
    wwx3 = w1 * wx2 - w2 * wx1
    xx1 = x1 + 2 * a * wx1 + 2 * wwx1
    xx2 = x2 + 2 * a * wx2 + 2 * wwx2
    xx3 = x3 + 2 * a * wx3 + 2 * wwx3
    return np.array((xx1, xx2, xx3))


def rot_euler_rodrigues_np(point, axis, angle):
    x = point
    a = np.cos(angle / 2)
    w = axis * np.sin(angle / 2)
    wx = np.cross(w, x)
    return x + 2 * a * wx + 2 * np.cross(w, wx)


rotation = rot_rodrigues


def main():
    import time
    count = 1000 * 1000
    angle = 0.5
    axes = np.random.random((count, 3))
    axes /= np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    points = np.random.random((count, 3))

    point = points[0]
    axis = np.array([0, 0, 1.])
    print(point)
    print()
    print(rot_rodrigues(point, axis, np.pi / 2))
    print(rot_rodrigues_np(point, axis, np.pi / 2))
    print(rot_euler_rodrigues(point, axis, np.pi / 2))
    print(rot_euler_rodrigues_np(point, axis, np.pi / 2))

    ret_raw = np.zeros((count, 3))
    start = time.time()
    for i, (point, axis) in enumerate(zip(points, axes)):
        ret_raw[i] = rot_rodrigues(point, axis, angle)
    print("duration: {}".format(time.time() - start))

    ret_er = np.zeros((count, 3))
    start = time.time()
    for i, (point, axis) in enumerate(zip(points, axes)):
        ret_er[i] = rot_euler_rodrigues(point, axis, angle)
    print("duration: {}".format(time.time() - start))

    print("error: {}".format(np.square(ret_raw - ret_er).sum()))


if __name__ == "__main__":
    main()
