__author__ = 's'

import unittest
import os
import numpy
import logging
import shutil

import algo.motion_estimation_sequence
import utils.image
import utils.logger_setup
from utils.flow import flow_equal
from time import time

try:
    import algo.fastdeepflow as fastdeepflow
except ImportError:
    raise Exception("Failed to load fastdeepflow module. Download from https://github.com/serge-m/deepflow")


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

        utils.logger_setup.setup_logging(default_level=logging.DEBUG)
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

    def test_pairs(self):
        templ_path_frm = os.path.join(path_dir_input, 'frame_{:04d}.png')
        idx_frame_start = 1
        idx_frame_end = 5

        img = [utils.image.imread(templ_path_frm.format(i)) for i in range(idx_frame_start, idx_frame_end)]

        me_parallel = algo.motion_estimation_sequence.MotionEstimationParallel()
        me_sequential = algo.motion_estimation_sequence.MotionEstimationSequential()
        pairs = [(img[1], img[0]), (img[1], img[0]), (img[2], img[1])]
        flow_parallel = me_parallel.calc_pairs(pairs)
        flow_sequential = me_sequential.calc_pairs(pairs)
        flow_gt = fastdeepflow.calc_flow(img[1], img[0])

        self.assertTrue(len(flow_parallel) == len(flow_sequential))
        self.assertTrue(numpy.array_equal(flow_gt, flow_sequential[0]))

        for i in range(0, len(flow_parallel)):
            self.assertTrue(numpy.array_equal(flow_parallel[i], flow_sequential[i]))

    def test_cached(self):
        logger = logging.getLogger(__name__)
        templ_path_frm = os.path.join(path_dir_input, 'frame_{:04d}.png')
        idx_frame_start = 1
        idx_frame_end = 6

        path_cache = os.path.join(path_result_dir, "cache.data")
        img = [utils.image.imread(templ_path_frm.format(i)) for i in range(idx_frame_start, idx_frame_end)]

        if os.path.exists(path_cache):
            os.remove(path_cache)

        self.assertFalse(os.path.exists(path_cache))

        me_parallel = algo.motion_estimation_sequence.MotionEstimationParallel()
        me_cached = algo.motion_estimation_sequence.MotionEstimationParallelCached(path_cahce=path_cache)

        start = time()
        flow_parallel = me_parallel.calc(img)
        time_parallel = time() - start
        logger.info("time_parallel {}".format(time_parallel))

        start = time()
        flow_cached1 = me_cached.calc(img[1:-1])
        time_cached1 = time() - start
        logger.info("time_cached1 {}".format(time_cached1))

        start = time()
        flow_cached2 = me_cached.calc(img)
        time_cached2 = time() - start
        logger.info("time_cached2 {}".format(time_cached2))

        start = time()
        flow_cached3 = me_cached.calc(img)
        time_cached3 = time() - start
        logger.info("time_cached3 {}".format(time_cached3))

        del me_cached
        me_cached2 = algo.motion_estimation_sequence.MotionEstimationParallelCached(path_cahce=path_cache)

        start = time()
        flow_cached4 = me_cached2.calc(img)
        time_cached4 = time() - start
        logger.info("time_cached4 {}".format(time_cached4))

        self.assertTrue(time_cached3 < 0.1 * time_cached2)
        self.assertTrue(time_cached3 < 0.1 * time_cached1)
        self.assertTrue(time_cached4 < 0.1 * time_cached1)
        self.assertTrue(flow_equal(flow_parallel, flow_cached2))
        self.assertTrue(flow_equal(flow_parallel, flow_cached3))
        self.assertTrue(flow_equal(flow_parallel, flow_cached4))
        self.assertTrue(flow_equal(flow_parallel[2:-1], flow_cached1[1:]))

