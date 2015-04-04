#!/usr/bin/python

import cv2

def imread(path):
    """Read image from file

    param: path path to source image

    return: image as numpy array
    """
    img = cv2.imread(path)

    if img is None:
        raise Exception("Unable to open '{}'".format(path))

    if img.ndim != 1 and img.ndim != 3:
        raise Exception("Unsupported number of dimensions")
        
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img[:,:,::-1]
        
    return img


def imwrite(path, img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, img)


def overlay(fg, bg, pos_x, pos_y):
    img = bg.copy()
    img[pos_y:pos_y+fg.shape[0], pos_x:pos_x+fg.shape[1]] = fg
    return img