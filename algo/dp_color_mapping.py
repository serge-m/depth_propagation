__author__ = 'Sergey Matyunin'
"""Draft version of depth propagation based on color mapping algorithm from [1]

[1] Li, Yanjie, Lifeng Sun, and Tianfan Xue. "Fast frame-rate up-conversion of depth video via video coding."
Proceedings of the 19th ACM international conference on Multimedia. ACM, 2011.
"""

from depth_propagation_base import DepthPropagationFwd
import cv2
import numpy

import scipy.ndimage.filters

import numba


def get_g():
    sigma = 3
    r = int(3*sigma)
    arr_in = numpy.zeros(shape=(2*r+1), dtype=numpy.float32)
    arr_in[r] = 1.
    arr_in = scipy.ndimage.filters.gaussian_filter1d(arr_in, sigma)
    return r, arr_in


def process_block(sbr, sbr_prime, b, b_prime, rg, g):
    all_c = sbr_prime.ravel()
    all_d = sbr.ravel()
    m = numpy.zeros(shape=(256,), dtype=numpy.int32)
    dv = numpy.zeros(shape=(256,), dtype=numpy.int32)

    for c, d in zip(all_c, all_d):
        dv[c] += d
        m[c] += 1

    dv = numpy.where(m != 0, (dv/m.astype(numpy.float32)).astype(numpy.int32), 0)

    all_c = b_prime.ravel()
    all_d = b.ravel()
    res = numpy.zeros_like(b)
    all_res = res.ravel()

    for idx in range(len(all_c)):
        c = all_c[idx]
        d = all_d[idx]
        dv_b = numpy.pad(dv, (rg,), 'constant', constant_values=(0.,))
        m_b = numpy.pad(m, (rg,), 'constant', constant_values=(0.,))

        start = c-rg+rg
        finish = c+rg+1+rg

        weights = (m_b[start:finish]!=0)*g
        depth = dv_b[start:finish]*weights

        sum_weights = weights.sum()
        if sum_weights > 0:
            d_res = depth.sum() / sum_weights
        else:
            d_res = d
        all_res[idx] = d_res
    return all_res.reshape(b.shape)


@numba.jit
def process_block_faster(sbr, sbr_prime, b, b_prime, rg, g):
    all_c = sbr_prime.ravel()
    all_d = sbr.ravel()
    m  = numpy.zeros(shape=(256+2*rg,), dtype=numpy.int32)
    dv = numpy.zeros(shape=(256+2*rg,), dtype=numpy.float32)

    for c, d in zip(all_c, all_d):
        dv[c+rg] += d
        m [c+rg] += 1

    dv = dv / numpy.maximum(m, 1)

    all_c = b_prime.ravel()
    all_d = b.ravel()
    res = numpy.zeros_like(b)
    all_res = res.ravel()

    for idx in range(len(all_c)):
        c = all_c[idx]
        d = all_d[idx]

        sum_weights = 0.
        sum_depth = 0.
        for j in range(0, 2*rg+1):
            w = (m[c+j] != 0) * g[j]
            sum_weights += w
            sum_depth += dv[c+j] * w

        all_res[idx] = sum_depth / sum_weights if sum_weights > 0 else d

    return all_res.reshape(b.shape)


def color_mapping(dpt_prev_w, frm_cur_gray, dpt_prev_w_b, frm_prev_w_gray_b, block_size, super_block_size, func_process_block=process_block_faster):
    res = numpy.zeros_like(dpt_prev_w)

    rg, g = get_g()

    for y in range(0, dpt_prev_w.shape[0], block_size):
        for x in range(0, dpt_prev_w.shape[1], block_size):

            y_start  = y #block_border+y-block_border
            y_finish = y + super_block_size

            x_start  = x
            x_finish = x + super_block_size

            sbr       = dpt_prev_w_b[y_start:y_finish, x_start:x_finish]
            sbr_prime = frm_prev_w_gray_b[y_start:y_finish, x_start:x_finish]
            b         = dpt_prev_w  [y:y+block_size, x:x+block_size]
            b_prime   = frm_cur_gray[y:y+block_size, x:x+block_size]

            t = func_process_block(sbr, sbr_prime, b, b_prime, rg=rg, g=g)
            res[y:y+block_size, x:x+block_size] = t
    return res


def get_part(img):
    #return img[250:300,350:400].astype('uint8')
    return img.astype('uint8')

class DPWithColorMapping(DepthPropagationFwd):
    def __init__(self, img, dpt, func_color_mapping=color_mapping, func_process_block=process_block_faster):
        super(DPWithColorMapping, self).__init__(img, dpt)
        self.func_color_mapping = func_color_mapping
        self.func_process_block = func_process_block


    def color_mapped_propagation_one_frame(self, i):
        u, v = self.flowFwd[i]

        dpt_prev, frm_prev, frm_cur = (self.res[i-1],
                                       self.img[i-1],
                                       self.img[i])

        dpt_prev_w = self.motion.warp(dpt_prev, u, v)
        frm_prev_w = self.motion.warp(frm_prev, u, v)

        (dpt_prev_w,
         frm_prev_w) = (get_part(i) for i in (dpt_prev_w,
                                              frm_prev_w))

        frm_cur_gray = cv2.cvtColor(frm_cur, cv2.COLOR_RGB2GRAY)
        frm_prev_w_gray = cv2.cvtColor(frm_prev_w, cv2.COLOR_RGB2GRAY)

        block_size = 4
        super_block_size = 8
        block_border = (super_block_size - block_size) / 2

        if block_size + 2 * block_border != super_block_size:
            raise Exception("Unsupported block size")

        for img_temp in (dpt_prev_w, frm_prev_w_gray):
            if img_temp.dtype != numpy.uint8:
                raise Exception("Unsupported image type. Use uint8")

        def add_borders(img):
            return cv2.copyMakeBorder(img,
                                      block_border,
                                      block_border,
                                      block_border,
                                      block_border,
                                      cv2.BORDER_REFLECT)

        dpt_prev_w_b = add_borders(dpt_prev_w)
        frm_prev_w_gray_b = add_borders(frm_prev_w_gray)

        self.logger.debug("color mapping started")
        res = self.func_color_mapping(dpt_prev_w,
                                      frm_cur_gray,
                                      dpt_prev_w_b,
                                      frm_prev_w_gray_b,
                                      block_size,
                                      super_block_size,
                                      func_process_block=self.func_process_block)
        self.logger.debug("color mapping done")
        self.res[i] = res

    def compensate_fwd(self):
        self.logger.debug("Motion compensation")
        for i in range(1, self.__len__()):
            self.color_mapped_propagation_one_frame(i)



