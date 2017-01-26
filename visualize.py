#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import os
import random

import numpy as np
import pygame
from pygame import locals as pgl
from scipy import misc


parser = argparse.ArgumentParser()
parser.add_argument("input_dir", metavar="INPUT_DIR")
args = parser.parse_args()

path = args.input_dir
assert os.path.isdir(path)
names = os.listdir(path)


def get_image():
    name = random.choice(names)
    ret = misc.imread(os.path.join(path, name), mode="RGB")
    return ret

image = get_image()

width, height, channels, pixel_size = 320, 240, 3, 5

pygame.init()
canvas = pygame.display.set_mode((width * pixel_size, height * pixel_size))
data = np.zeros((height, pixel_size, width, pixel_size, channels))
view = data.reshape((height * pixel_size, width * pixel_size, channels)).transpose((1, 0, 2))
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

    pygame.surfarray.blit_array(canvas, view)
    pygame.display.update()
