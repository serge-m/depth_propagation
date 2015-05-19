#!/usr/bin/python
__author__ = 's'

import logging

import motion_estimation_sequence


class DepthPropagation(object):
    """
    Base class for depth propagation algorithms.
    """

    def __init__(self):
        self.res = []

    def preprocess(self):
        """
        Some preprocessing, possible computation intensive
        :return:
        """
        pass

    def __getitem__(self, key):
        """
        Return result of computation using operator[]
        :param key: frame index, starting from 0
        :return:
        """
        return self.res[key]

    def __len__(self):
        """
        Length of image sequence
        :return:
        """
        return len(self.res)

    def get_propagation_distance(self, idx_frame):
        """
        Some distance measure betweek key frame(s) and frame idx_frame
        :param idx_frame:
        :return:
        """
        raise Exception("Not implemented")


class DepthPropagationFwd(DepthPropagation):
    def __init__(self, img, dpt):
        if dpt[0] is None:
            raise Exception("First depth map frame must be defined")

        super(DepthPropagationFwd, self).__init__()

        self.res = [None,] * len(img)
        self.img = img
        self.dpt = dpt

        self.res[0] = self.dpt[0]

        if len(self.img) != len(self.dpt):
            raise Exception("Length of input image sequence must match length of input depth sequence."
                            "Use empty (None) placeholders for unknown depth frames")

        self.logger = logging.getLogger(__name__)

        self.flowFwd = None
        # self.motion = motion_estimation.MotionEstimation()
        self.motion = motion_estimation_sequence.MotionEstimationParallelCached()

    def preprocess_flow(self):
        self.logger.debug("Optcal flow calculation")
        if self.flowFwd is None:

            self.flowFwd = self.motion.calc(self.img)

    def compensate_fwd(self):
        self.logger.debug("Motion compensation")
        for i in range(1, self.__len__()):
            u, v = self.flowFwd[i]
            self.res[i] = self.motion.warp(self.res[i-1], u, v)

    def preprocess(self):
        self.preprocess_flow()
        self.compensate_fwd()

    def get_propagation_distance(self, idx_frame):
        """
        Calculates distance (or some alternative) from key frames to frame with index idx_frame
        :param idx_frame:
        :return:
        """
        return abs(0-idx_frame)


class DepthPropagationBwd(DepthPropagation):
    def __init__(self, img, dpt, dp_class_fwd=DepthPropagationFwd):
        if dpt[-1] is None:
            raise Exception("Last depth map frame must be defined")

        super(DepthPropagationBwd, self).__init__()
        self.dp_impl = DepthPropagationFwd(img[::-1], dpt[::-1])

    def __getitem__(self, key):
        return self.dp_impl[self.__len__()-1-key]

    def preprocess(self):
        return self.dp_impl.preprocess()

    def __len__(self):
        return len(self.dp_impl)

    def get_propagation_distance(self, idx_frame):
        """
        Calculates distance (or some alternative) from key frames to frame with index idx_frame
        :param idx_frame:
        :return:
        """
        return abs(self.__len__()-1-idx_frame)
