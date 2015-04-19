__author__ = 's'

import unittest
import os
import numpy
import logging
import shutil

import algo.motion_estimation_sequence
import utils.image


path_dir_cur = os.path.dirname(__file__)
path_dir_input = os.path.join(path_dir_cur, 'data/input/sintel/final/alley_2/', '')
path_result_dir = os.path.join(path_dir_cur, 'data/results/me_sequence', '')
path_reference_dir = os.path.join(path_dir_cur, 'data/reference', '')


def mkdirs(path, mode=0777, exist_ok=False):
    if not os.path.exists(path) or not exist_ok:
        os.makedirs(path, mode=mode)


class TestMotionEstimationSequence(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMotionEstimationSequence, self).__init__(*args, **kwargs)

        logger = logging.getLogger(__name__)

        if os.path.exists(path_result_dir):
            shutil.rmtree(path_result_dir)
            logger.info("{} directory cleaned".format(path_result_dir))
        if not os.path.exists(path_reference_dir):
            raise Exception("Failed to locate reference directory")
        mkdirs(path_result_dir, exist_ok=False)

    def test_parallel(self):
        templ_path_frm = os.path.join(path_dir_input, 'frame_{:04d}.png')
        idx_frame_start = 1
        idx_frame_end = 5

        img = [utils.image.imread(templ_path_frm.format(i)) for i in range(idx_frame_start, idx_frame_end)]

        me_parallel = algo.motion_estimation_sequence.MotionEstimationParallel()
        me_sequential = algo.motion_estimation_sequence.MotionEstimationSequential()
        flow_parallel = me_parallel.calc(img)
        flow_sequential = me_sequential.calc(img)

        self.assertTrue(len(flow_parallel) == len(flow_sequential))
        self.assertTrue(flow_parallel[0] is None)

        for i in range(1, len(flow_parallel)):
            self.assertTrue(numpy.array_equal(flow_parallel[i], flow_sequential[i]))