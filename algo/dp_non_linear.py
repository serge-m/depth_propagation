__author__ = 'Sergey Matyunin'

import algo
import cv2
import numba

import numpy


class DepthPropagationNonLinear(algo.DepthPropagationFwd):
    def __init__(self, img, dpt, radius_local=2):
        super(DepthPropagationNonLinear, self).__init__(img, dpt)
        self.radius_local = radius_local

    def dp_non_linear_bordered(self, frm_cur, dpt_prev_w, frm_prev_w):
        radius = self.radius_local
        self.logger.debug("radius {}".format(radius))
        self.logger.debug("frm_cur {}".format(frm_cur.shape))

        dpt_dst = numpy.zeros_like(dpt_prev_w)
        height, width = self.img[0].shape[0:2]
        arr_candidates = numpy.zeros(shape=(2*radius+1)**2, dtype=numpy.int32)
        for idx_h in xrange(radius, radius + height):
            for idx_w in xrange(radius, radius + width):
                arr_candidates[:] = 0
                count_candidates = 0
                for idx_k_h in xrange(idx_h - radius, idx_h + radius + 1):
                    for idx_k_w in xrange(idx_w - radius, idx_w + radius + 1):
                        arr_candidates[count_candidates] = dpt_prev_w[idx_k_h, idx_k_w]
                        count_candidates += 1

                dpt_dst[idx_h, idx_w] = numpy.median(arr_candidates[:count_candidates])

        return dpt_dst[radius:-radius, radius:-radius]

    def dp_non_linear_by_frame_index(self, i):
        u, v = self.flowFwd[i]

        convert_to_uint8 = True

        frm_cur = self.img_yuv[i]
        dpt_prev_w = self.motion.warp(self.res[i - 1], u, v)
        frm_prev_w = self.motion.warp(self.img_yuv[i - 1], u, v)

        if convert_to_uint8:
            dpt_prev_w = dpt_prev_w.astype(numpy.uint8)
            frm_prev_w = frm_prev_w.astype(numpy.uint8)

        (frm_cur,
         dpt_prev_w,
         frm_prev_w) = map(lambda im: self.add_borders(im, self.radius_local),
                           (frm_cur,
                            dpt_prev_w,
                            frm_prev_w))

        for img_temp in (dpt_prev_w, frm_prev_w):
            self.logger.info("{}".format(img_temp.dtype))
            if img_temp.dtype != numpy.uint8:
                raise Exception("Unsupported image type. Use uint8")

        self.res[i] = self.dp_non_linear_bordered(frm_cur, dpt_prev_w, frm_prev_w)

    def add_borders(self, img, size_border):
        return cv2.copyMakeBorder(img,
                                  size_border,
                                  size_border,
                                  size_border,
                                  size_border,
                                  cv2.BORDER_REPLICATE)

    def compensate_fwd(self):
        self.logger.debug("Prepare YUV representation")

        self.img_yuv = map(lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2YUV),
                           self.img)

        self.logger.debug("Motion compensation")

        for i in range(1, self.__len__()):
            self.dp_non_linear_by_frame_index(i)
