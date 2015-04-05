#!/usr/bin/python

import unittest
import algo.motion_estimation
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


class TestMotionEstimation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMotionEstimation, self).__init__(*args, **kwargs)

        logger = logging.getLogger(__name__)

        if os.path.exists(path_result_dir):
            shutil.rmtree(path_result_dir)
            logger.info("{} directory cleaned".format(path_result_dir))
        if not os.path.exists(path_reference_dir):
            raise Exception("Failed to locate reference directory")
        mkdirs(path_result_dir, exist_ok=False)


    def test_image_loading(self):
        path_dir_data = os.path.join(path_dir_input, 'artificial/50x25', '')

        path_img = path_dir_data + '50x20_frm_00000.png'
        img_g = cv2.imread(path_img)[:,:,::-1]
        img_0 = utils.image.imread(path_img)
        self.assertTrue(numpy.array_equal(img_0, img_g), msg="image reading is failed {}".format(path_img))


    def test_image_generation(self):
        path_dir_data = os.path.join(path_dir_input, 'artificial/50x25', '')

        for i in range(3):
            path_img_g = path_dir_data + '50x20_frm_{:05d}.png'.format(i)
            path_img_0 = path_dir_data + 'im_{:05d}.png'.format(i)
            img_g = utils.image.imread(path_img_g)
            img_0 = utils.image.imread(path_img_0)
            self.assertTrue(numpy.array_equal(img_0, img_g), msg="generation for {} is wrong".format(path_img_g))


    def test_of(self):
        path_dir_data = os.path.join(path_dir_input, 'artificial/50x25', '')

        img0 = utils.image.imread(path_dir_data + '50x20_frm_00000.png')
        img1 = utils.image.imread(path_dir_data + '50x20_frm_00001.png')

        size = (img0.shape[0]*4, img0.shape[1]*4)
        img0 = cv2.resize(img0, size)
        img1 = cv2.resize(img1, size)


        me = algo.motion_estimation.MotionEstimation()

        u, v = me.calc(img0, img1)

        img0warped = me.warp(img0, u, v)
        img1warped = me.warp(img1, u, v)

        d01 = numpy.sum(numpy.abs(img0-img1.astype('float32')))
        d0w1= numpy.sum(numpy.abs(img0warped-img1))
        d01w= numpy.sum(numpy.abs(img0-img1warped))
        print "d01 {}, d0w1 {}, d01w {}".format(d01, d0w1, d01w)
        self.assertTrue(d01w<d01)
        self.assertTrue(d0w1>d01)


    def test_of_sintel(self):
        path_dir_frm = os.path.join(path_dir_input, 'sintel/final/alley_1/', '')
        path_dir_dpt = os.path.join(path_dir_input, 'sintel/depth_viz/alley_1/', '')

        img0 = utils.image.imread(path_dir_frm + 'frame_0001.png')
        img1 = utils.image.imread(path_dir_frm + 'frame_0002.png')

        me = algo.motion_estimation.MotionEstimation()

        u, v = me.calc(img0, img1)

        img0warped = me.warp(img0, u, v)
        img1warped = me.warp(img1, u, v)

        path_dst = os.path.join(path_result_dir, 'alley_1', '')
        path_reference = os.path.join(path_reference_dir, 'alley_1', '')
        mkdirs(path_dst, exist_ok=True)

        utils.image.imwrite(os.path.join(path_dst, 'img0.png'), img0)
        utils.image.imwrite(os.path.join(path_dst, 'img1.png'), img1)
        utils.image.imwrite(os.path.join(path_dst, 'img0warped.png'), img0warped)
        utils.image.imwrite(os.path.join(path_dst, 'img1warped.png'), img1warped)

        img1warped_r = utils.image.imread(os.path.join(path_reference, 'img1warped.png'))

        self.assertTrue(numpy.allclose(img1warped, img1warped_r, atol=0.5))

        d01 = numpy.sum(numpy.abs(img0-img1.astype('float32')))
        d0w1= numpy.sum(numpy.abs(img0warped-img1))
        d01w= numpy.sum(numpy.abs(img0-img1warped))
        print "d01 {}, d0w1 {}, d01w {}".format(d01, d0w1, d01w)
        self.assertTrue(d01w<d01)
        self.assertTrue(d0w1>d01)

if __name__ == '__main__':
    unittest.main()
