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
