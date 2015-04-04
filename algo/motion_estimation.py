#!/usr/bin/python

try:
    import fastdeepflow
except ImportError:
    raise Exception("Failed to load fastdeepflow module. Download from https://github.com/serge-m/deepflow")
    fastdeepflow = None


class MotionEstimation:
    def __init__(self):
        if fastdeepflow is None:
            raise Exception("Failed to load optical flow library")


    def calc(self, image_cur, image_ref):
        return fastdeepflow.calc_flow(image_cur, image_ref)


    def warp(self, image, flow_x, flow_y):
        return fastdeepflow.warp_image(image, flow_x, flow_y)


