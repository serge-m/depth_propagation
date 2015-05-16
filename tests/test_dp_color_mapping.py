#!/usr/bin/python
__author__ = 'Sergey Matyunin'


import unittest
import algo.motion_estimation
import utils.image

import os
import numpy
import cv2
import logging

import shutil
import algo
import algo.dp_color_mapping
from time import time
import utils.flow
import utils.logger_setup
import logging


path_dir_cur = os.path.dirname(__file__)
path_dir_input = os.path.join(path_dir_cur, 'data/input', '')
path_result_dir = os.path.join(path_dir_cur, 'data/results/dp_color_mapping', '')
path_reference_dir = os.path.join(path_dir_cur, 'data/reference', '')

def mkdirs(path, mode=0777, exist_ok=False):
    if not os.path.exists(path) or not exist_ok:
        os.makedirs(path, mode=mode)


class TestDPColorMapping(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDPColorMapping, self).__init__(*args, **kwargs)

        utils.logger_setup.setup_logging(default_path='../logger_config.json',
                                         default_level=logging.DEBUG,
                                         error_if_config_missing=True)
        logger = logging.getLogger(__name__)

        if os.path.exists(path_result_dir):
            shutil.rmtree(path_result_dir)
            logger.info("{} directory cleaned".format(path_result_dir))
        if not os.path.exists(path_reference_dir):
            raise Exception("Failed to locate reference directory")
        mkdirs(path_result_dir, exist_ok=False)

    def test(self):
        logger = logging.getLogger(__name__)

        templ_path_frm = os.path.join(path_dir_input, 'sintel/final/alley_2/'    , 'frame_{:04d}.png')
        templ_path_dpt = os.path.join(path_dir_input, 'sintel/depth_viz/alley_2/', 'frame_{:04d}.png')
        idx_frame_start = 1
        idx_frame_end = 11

        templ_path_dst = os.path.join(path_result_dir, 'dp_{:05d}.png')

        def load_and_crop(path):
            img = utils.image.imread(path)
            img = img[150:350,250:400,]
            return img

        img = [load_and_crop(templ_path_frm.format(i)) for i in range(idx_frame_start, idx_frame_end)]
        dpt = [None for i in range(idx_frame_start, idx_frame_end)]
        dpt_gt = [cv2.cvtColor(load_and_crop(templ_path_dpt.format(i)), cv2.COLOR_RGB2GRAY) for i in
                  range(idx_frame_start, idx_frame_end)]

        dpt[0] = cv2.cvtColor(load_and_crop(templ_path_dpt.format(idx_frame_start)), cv2.COLOR_RGB2GRAY)


        dp = algo.dp_color_mapping.DPWithColorMapping(img, dpt, )
        dp0 = algo.DepthPropagationFwd(img, dpt)
        dp_bwd = algo.DepthPropagationBwd(img[::-1], dpt[::-1])

        start = time()
        dp.preprocess()
        logger.info("Time dp color mapping {}".format(time()-start))

        start = time()
        dp0.preprocess()
        logger.info("Time dp0 {}".format(time()-start))

        start = time()
        dp_bwd.preprocess()
        logger.info("Time db_bwd {}".format(time()-start))

        self.assertTrue(utils.flow.flow_equal(dp0[::-1], dp_bwd))

        idx_frame = 2
        gt = dpt_gt[idx_frame]
        version1 = dp[idx_frame]
        version2 = dp0[idx_frame]
        diff1 = utils.image.abs_diff(version1, gt)
        diff2 = utils.image.abs_diff(version2, gt)
        logger.info("SUM Diff 1 {}, Diff 2 {}".format(diff1.sum(), diff2.sum()))
        logger.info("MAX Diff 1 {}, Diff 2 {}".format(diff1.max(), diff2.max()))

        templ_dp_color_mapping_ref = os.path.join(path_reference_dir,"fwd/block_default/dp_{:05d}.png")
        dp_loaded = [cv2.cvtColor(utils.image.imread(templ_dp_color_mapping_ref.format(i)), cv2.COLOR_RGB2GRAY) for i in
                  range(0, 10)]

        for i in range(len(dp)):
            d = utils.image.abs_diff(dp[idx_frame], dp_loaded[idx_frame])
            logger.info("Frame {}, max {}, sum {}".format(i, d.max(), d.sum()))
            self.assertTrue(d.max() < 10)


        # compare with faster version
        templ_dp_color_mapping_ref = os.path.join(path_reference_dir,"fwd/block_default_faster/dp_{:05d}.png")
        dp_loaded = [cv2.cvtColor(utils.image.imread(templ_dp_color_mapping_ref.format(i)), cv2.COLOR_RGB2GRAY) for i in
                  range(0, 10)]

        for i in range(len(dp)):
            d = utils.image.abs_diff(dp[idx_frame], dp_loaded[idx_frame])
            logger.info("Frame {}, max {}, sum {}".format(i, d.max(), d.sum()))
            self.assertTrue(d.max() < 5)


if __name__ == '__main__':
    unittest.main()
