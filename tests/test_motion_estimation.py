#!/usr/bin/python

import unittest
import algo.motion_estimation
import utils.image

import os
import numpy
import cv2



class TestMotionEstimation(unittest.TestCase):
    def test_image_loading(self):
        path_dir_data = os.path.join(os.path.dirname(__file__), 'data/artificial/50x25', '')

        path_img = path_dir_data + '50x20_frm_00000.png'
        img_g = cv2.imread(path_img)[:,:,::-1]
        img_0 = utils.image.imread(path_img)
        self.assertTrue(numpy.array_equal(img_0, img_g), msg="image reading is failed {}".format(path_img))

    def test_image_generation(self):
        path_dir_data = os.path.join(os.path.dirname(__file__), 'data/artificial/50x25', '')

        for i in range(3):
            path_img_g = path_dir_data + '50x20_frm_{:05d}.png'.format(i)
            path_img_0 = path_dir_data + 'im_{:05d}.png'.format(i)
            img_g = utils.image.imread(path_img_g)
            img_0 = utils.image.imread(path_img_0)
            self.assertTrue(numpy.array_equal(img_0, img_g), msg="generation for {} is wrong".format(path_img_g))

    def test_of(self):
        path_dir_data = os.path.join(os.path.dirname(__file__), 'data/artificial/50x25', '')

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

if __name__ == '__main__':
    unittest.main()
