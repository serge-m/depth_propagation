#!/usr/bin/python
__author__ = 's'

import logging

import motion_estimation_sequence


class DepthPropagation(object):
    def __init__(self, img, dpt):
        self.length = len(img)
        self.img = img
        self.dpt = dpt
        self.res = [None,] * self.length
        if self.length != len(self.dpt):
            raise Exception("Length of input image sequence must match length of input depth sequence."
                            "Use empty (None) placeholders for unknown depth frames")


    def preprocess(self):
        pass


    def __getitem__(self, key):
        return self.res[key]




class DepthPropagationFwd(DepthPropagation):
    def __init__(self, img, dpt):
        super(DepthPropagationFwd, self).__init__(img, dpt)
        self.logger = logging.getLogger(__name__)
        self.res[0] = self.dpt[0]
        self.flowFwd = None
        # self.motion = motion_estimation.MotionEstimation()
        self.motion = motion_estimation_sequence.MotionEstimationParallelCached()

    def preprocess_flow(self):
        self.logger.debug("Optcal flow calculation")
        if self.flowFwd is None:

            self.flowFwd = self.motion.calc(self.img)

    def compensate_fwd(self):
        self.logger.debug("Motion compensation")
        for i in range(1, self.length):
            u, v = self.flowFwd[i]
            self.res[i] = self.motion.warp(self.res[i-1], u, v)

    def preprocess(self):
        self.preprocess_flow()
        self.compensate_fwd()


