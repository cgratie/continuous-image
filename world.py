#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import os
import random

from keras.layers import Input, Flatten, Dense, Reshape, Dropout, Convolution2D, MaxPooling2D, Permute, AveragePooling2D, AtrousConv2D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import pygame
from pygame import locals as pgl

from eyes import GridEye
from objects import ImageObj


def main():
    # get input folder with images
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", metavar="INPUT_DIR")
    args = parser.parse_args()

    # initialize available files
    path = args.input_dir
    assert os.path.isdir(path)
    names = os.listdir(path)

    # setup parameters
    eye_width, eye_height = 64, 64
    eye_psize = 8
    img_width, img_height = 80, 60
    img_psize, img_dsize = 8, 4
    margin = 5
    channels = 3
    xy_step, angle_step = 1, 0.1

    # setup objects
    eye = GridEye(pos=(0, 0, 100), up=(0, 1, 0), front=(0, 0, -1),
                  width=eye_width, height=eye_height, size=1, fdist=-25)
    obj = ImageObj(pos=(0, 0, 0), up=(0, 1, 0), front=(0, 0, -1),
                   path=os.path.join(path, random.choice(names)))
    world = [obj]

    # setup learning
    conv_args = dict(activation="tanh", bias=False)
    z = x = Input(shape=(eye_width, eye_height, channels))
    z = Permute(dims=(3, 1, 2))(z)
    z = AtrousConv2D(3 * 4 ** 5, 2, 2, atrous_rate=(32, 32), **conv_args)(z)
    z = AtrousConv2D(3 * 4 ** 4, 2, 2, atrous_rate=(16, 16), **conv_args)(z)
    z = AtrousConv2D(3 * 4 ** 5, 2, 2, atrous_rate=(8, 8), **conv_args)(z)
    z = AtrousConv2D(3 * 4 ** 4, 2, 2, atrous_rate=(4, 4), **conv_args)(z)
    z = AtrousConv2D(3 * 4 ** 5, 2, 2, atrous_rate=(2, 2), **conv_args)(z)
    z = AtrousConv2D(80 * 60 * 3, 2, 2, atrous_rate=(1, 1), activation="linear", bias=False)(z)
    z = Reshape(target_shape=(img_width, img_height, channels))(z)
    model = Model(input=x, output=z)
    model.compile(optimizer=Adam(lr=1e-5), loss="mse", metrics=None)
    model.summary()

    # setup canvas
    win_width = eye_width * eye_psize + img_width * img_psize + 3 * margin
    win_height = max(2 * img_height * img_psize + 3 * margin, 2 * margin + eye_height * eye_psize)
    pygame.init()
    canvas = pygame.display.set_mode((win_width, win_height))
    data = np.full((win_width, win_height, channels), 128, dtype="uint8")
    eye_view = data[margin:margin + eye_width * eye_psize, margin:margin + eye_height * eye_psize].reshape(
        eye_width, eye_psize, eye_height, eye_psize, channels
    )
    img_view = data[2 * margin + eye_width * eye_psize:2 * margin + eye_width * eye_psize + img_width * img_psize,
                    margin:margin + img_height * img_psize].reshape(
        img_width, img_psize, img_height, img_psize, channels
    )
    mem_view = data[2 * margin + eye_width * eye_psize:2 * margin + eye_width * eye_psize + img_width * img_psize,
                    2 * margin + img_height * img_psize:2 * margin + 2 * img_height * img_psize].reshape(
        img_width, img_psize, img_height, img_psize, channels
    )

    move = None
    rot = None
    ctrl = False
    refresh = False
    eye_changed = True
    img_changed = True
    while True:
        for event in pygame.event.get():
            if event.type == pgl.KEYUP:
                if not ctrl:
                    move = None
                    rot = None
                    refresh = False
                if event.key == pgl.K_LCTRL:
                    ctrl = False
            if event.type == pgl.KEYDOWN:
                if event.key == pgl.K_ESCAPE:
                    pygame.quit()
                    exit()
                elif event.key == pgl.K_LCTRL:
                    ctrl = True
                elif event.key == pgl.K_SPACE:
                    refresh = True
                elif event.key == pgl.K_LEFT:
                    move = (-xy_step, 0, 0, True) if not ctrl else None
                    rot = ('u', angle_step) if ctrl else None
                elif event.key == pgl.K_RIGHT:
                    move = (xy_step, 0, 0, True) if not ctrl else None
                    rot = ('u', -angle_step) if ctrl else None
                elif event.key == pgl.K_UP:
                    move = (0, xy_step, 0, True) if not ctrl else None
                    rot = ('r', angle_step) if ctrl else None
                elif event.key == pgl.K_DOWN:
                    move = (0, -xy_step, 0, True) if not ctrl else None
                    rot = ('r', -angle_step) if ctrl else None
                elif event.key == pgl.K_c:
                    move = None
                    rot = ('f', angle_step)
                elif event.key == pgl.K_x:
                    move = None
                    rot = ('f', -angle_step)
                elif event.key == pgl.K_a:
                    move = (0, 0, xy_step, True)
                    rot = None
                elif event.key == pgl.K_z:
                    move = (0, 0, -xy_step, True)
                    rot = None
                else:
                    move = None
                    rot = None

                    if event.key == pgl.K_b:
                        obj.mode = "bilinear"
                    elif event.key == pgl.K_n:
                        obj.mode = "nearest"

                    if event.key == pgl.K_p:
                        eye.fdist = None
                    elif event.key == pgl.K_f:
                        eye.fdist = 25
                    elif event.key == pgl.K_r:
                        eye.fdist = -25

        if move is not None:
            eye.move(*move)
            eye_changed = True
        if rot is not None:
            eye.rot(*rot)
            eye_changed = True

        if refresh:
            obj.set_image(os.path.join(path, random.choice(names)))
            img_changed = True

        if img_changed:
            y = (obj.image / 255).reshape((img_width, img_dsize,
                                           img_height, img_dsize, channels)).mean(axis=(1, 3))
            img_view[...] = (255 * y).clip(0, 255).astype("uint8")[:, None, :, None, :]

        if eye_changed or img_changed:
            eye_view[...] = eye.see(world)[:, None, :, None, :]

        if img_changed or eye_changed:
            model.fit(eye.see(world)[None], y[None], batch_size=1, nb_epoch=1)
            mem_view[...] = (255 * model.predict(eye.see(world)[None])[0, :, None, :, None, :]).clip(0, 255).astype('uint8')
            pygame.surfarray.blit_array(canvas, data)
            pygame.display.update()

        eye_changed = False
        img_changed = False


if __name__ == "__main__":
    main()
