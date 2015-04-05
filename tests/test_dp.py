#!/usr/bin/python

import unittest
import algo
import utils.image

import os
import numpy
import cv2
import logging

import shutil


path_dir_cur = os.path.dirname(__file__)
path_dir_input = os.path.join(path_dir_cur, 'data/input', '')
path_result_dir = os.path.join(path_dir_cur, 'data/results', '')
path_reference_dir = os.path.join(path_dir_cur, 'data/reference', '')


def mkdirs(path, mode=0777, exist_ok=False):
    if not os.path.exists(path) or not exist_ok:
        os.makedirs(path, mode=mode)


def abs_diff(img0, img1):
    return numpy.abs(img0-img1.astype('float32'))


class TestDepthPropagation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDepthPropagation, self).__init__(*args, **kwargs)

        logger = logging.getLogger(__name__)

        if os.path.exists(path_result_dir):
            shutil.rmtree(path_result_dir)
            logger.info("{} directory cleaned".format(path_result_dir))
        if not os.path.exists(path_reference_dir):
            raise Exception("Failed to locate reference directory")
        mkdirs(path_result_dir, exist_ok=False)

        self.prepared_dp_tests = False
        self.img = self.dpt = self.dp = None

        self.path_reference = os.path.join(path_reference_dir, 'alley_1/dp', '')
        self.path_dst = os.path.join(path_result_dir, 'alley_1/dp', '')
        mkdirs(self.path_dst, exist_ok=True)


    def prepare_dp_tests_(self):
        if self.prepared_dp_tests:
            return
        templ_path_frm = os.path.join(path_dir_input, 'sintel/final/alley_1/'    , 'frame_{:04d}.png')
        templ_path_dpt = os.path.join(path_dir_input, 'sintel/depth_viz/alley_1/', 'frame_{:04d}.png')

        self.idx_frame_start = 1
        self.idx_frame_end = 11




        self.img = [utils.image.imread(templ_path_frm.format(i)) for i in range(self.idx_frame_start, self.idx_frame_end)]
        self.dpt = [None for i in range(self.idx_frame_start, self.idx_frame_end)]
        self.dpt_gt = [utils.image.imread(templ_path_dpt.format(i)) for i in range(self.idx_frame_start, self.idx_frame_end)]
        self.dpt_gt = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in self.dpt_gt]

        self.dpt[0] = cv2.cvtColor(utils.image.imread(templ_path_dpt.format(self.idx_frame_start)), cv2.COLOR_RGB2GRAY)

        self.dp = algo.DepthPropagationFwd(self.img, self.dpt)
        self.dp.preprocess()

        self.prepared_dp_tests = True


    def test_dp_sintel(self):
        self.prepare_dp_tests_()

        templ_path_dst = os.path.join(self.path_dst, 'frame_{:04d}.png')
        templ_path_dst_diff_orig = os.path.join(self.path_dst, 'frame_diff_orig_{:04d}.png')
        templ_path_dst_diff_dp   = os.path.join(self.path_dst, 'frame_diff_dp_{:04d}.png')
        templ_path_reference = os.path.join(self.path_reference, 'frame_{:05d}.png')

        self.assertTrue(self.dpt[0].shape == (436, 1024))

        for i in range(self.dp.length):
            utils.image.imwrite(templ_path_dst.format(i), self.dp[i])
            ref = cv2.cvtColor(utils.image.imread(templ_path_reference.format(i)), cv2.COLOR_RGB2GRAY)
            self.assertTrue(numpy.allclose(ref, self.dp[i], atol=0.5),
                            msg="Failed comparison for image #{}({})".format(i, i+self.idx_frame_start))

        for i in range(self.dp.length):
            diff_dp   = abs_diff(self.dpt_gt[i], self.dp[i])
            diff_orig = abs_diff(self.dpt_gt[i], self.dp[0])
            utils.image.imwrite(templ_path_dst_diff_dp.format(i), diff_dp)
            utils.image.imwrite(templ_path_dst_diff_orig.format(i), diff_orig)

            self.assertTrue(numpy.sum(diff_orig) >= numpy.sum(diff_dp),
                            msg="Difference for frame {} must be less for DP version".format(i))


if __name__ == '__main__':
    unittest.main()