__author__ = 'Sergey Matyunin'
import numpy
import algo.fastdeepflow
import logging
from time import time
import cv2
import matplotlib.pyplot as plt
import algo.motion_analysis.occlusion_detection as od

block_size = 20


def block_mv_median_for_occlusions(flow_bwd, map_occl):
    flow_bwd = map(lambda x: x.copy(), flow_bwd)
    h, w = flow_bwd[0].shape
    for h_start in xrange(0, h, block_size):
        for w_start in xrange(0, w, block_size):

            def get_patch(img):
                return img[h_start:h_start+block_size, w_start:w_start+block_size]

            map_occl_block = get_patch(map_occl)
            flow_bwd_block = map(get_patch, flow_bwd)
            u1, v1 = flow_bwd_block

            if numpy.any(map_occl_block):
                u1[map_occl_block] = numpy.median(u1[map_occl_block])
                v1[map_occl_block] = numpy.median(v1[map_occl_block])

    return flow_bwd


def warp_image(img0, u, v):
    if img0.ndim == 3:
        n_channels = img0.shape[-1]
        if n_channels in [1,3]:
            return algo.fastdeepflow.warp_image(img0, u, v)
        else:
            res = [algo.fastdeepflow.warp_image(img0[:, :, idx_plane], u, v) for idx_plane in xrange(n_channels)]
            return numpy.dstack(res)
    else:
        return algo.fastdeepflow.warp_image(img0, u, v)


def prepare_candidates(img0, flow_fwd, flow_bwd, map_occl):
    logger = logging.getLogger(__name__)
    start = time()
    flow_bwd_filt = block_mv_median_for_occlusions(flow_bwd, map_occl=map_occl)
    logger.info("block_mv_median_for_occlusions time {}".format(time()-start))

    list_coeffs = numpy.array([[1,1], [0.5,0.5], [-1,-1], [-0.5,-0.5]], dtype=numpy.float32)

    list_img0_warped = [warp_image(img0, *flow_fwd),]

    for coeff in list_coeffs:
        flow_bwd_filt_coeff = coeff[0]*flow_bwd_filt[0], coeff[1]*flow_bwd_filt[1]
        list_img0_warped.append(warp_image(img0, *flow_bwd_filt_coeff))
    return list_img0_warped


def block_histo_choose(map_occl, img1, list_img0_warped, list_dpt0_warped, th_num_occl):
    logger = logging.getLogger(__name__)
    dpt_res = numpy.zeros_like(list_dpt0_warped[0])

    h, w, _ = img1.shape

    for h_start in xrange(0*block_size, h, block_size):
        for w_start in xrange(0*block_size, w, block_size):
            def get_patch(img):
                return img[h_start:h_start+block_size, w_start:w_start+block_size]

            img1_block = get_patch(img1)

            list_img0_warped_block = map(get_patch, list_img0_warped)
            map_occl_block = get_patch(map_occl)

            if numpy.count_nonzero(map_occl_block) < th_num_occl:
                get_patch(dpt_res)[:, :] = get_patch(list_dpt0_warped[0])
                continue

            if 0:
                plt.figure()
                plt.imshow(map_occl_block)
                plt.figure()
                plt.imshow(img1_block)

                plt.figure()
                for idx, bl in enumerate(list_img0_warped_block):
                    plt.subplot(2, 2, idx+1)
                    plt.imshow(bl.astype('uint8'))

            def calc_hist(img0_block):
                imgs = [img0_block[:, :, i].astype('uint8') for i in range(3)]
                hist = [cv2.calcHist([img], [0, ],  map_occl_block.astype('uint8'), [8,], [0, 256], ) for img in imgs]
                hist = numpy.vstack(hist)
                hist = cv2.normalize(hist)
                return hist

            hist = calc_hist(img1_block)
            list_hist = map(calc_hist, list_img0_warped_block)

            if 0:
                plt.figure()
                plt.plot(hist.ravel(), )
    #             plt.figure(figsize=(20,10))
                for idx, hi in enumerate(list_hist):
    #                 plt.subplot(2,2, idx+1)
                    plt.plot(hi.ravel(), '+x.*'[idx])
        #                 print "d", numpy.abs(hist-hi).sum()
        #                 print list_hist[idx]

            list_dist = map(lambda hist_tmp: cv2.compareHist(hist_tmp, hist, cv2.cv.CV_COMP_BHATTACHARYYA), list_hist)
            logger.debug("list_dist {}".format(list_dist))
            idx_best = numpy.argmin(list_dist)

            get_patch(dpt_res)[:, :] = get_patch(list_dpt0_warped[idx_best])
        #     distances = map(lambda warped_block: )
        #     break
        # break

    return dpt_res, None


def calc_covered(flow_bwd):
    logger = logging.getLogger(__name__)
    start = time()
    covered = od.get_covered(*flow_bwd)
    logger.debug("Time_get_covered {}".format(time()-start))

    covered = numpy.clip(covered, 0, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(covered, cv2.MORPH_OPEN, kernel)
    return morph


def compensate_and_fix_occlusions(img0, img1, flow_fwd, flow_bwd, dpt0, th_occl):

    occl = calc_covered(flow_bwd=flow_bwd)
    map_occl = occl < th_occl

    list_img0_warped = prepare_candidates(img0, flow_fwd=flow_fwd, flow_bwd=flow_bwd, map_occl=map_occl)
    list_dpt0_warped = prepare_candidates(dpt0, flow_fwd=flow_fwd, flow_bwd=flow_bwd, map_occl=map_occl)

    dpt_res, _ = block_histo_choose(map_occl=map_occl, img1=img1,
                                     list_img0_warped=list_img0_warped,
                                     list_dpt0_warped=list_dpt0_warped,
                                     th_num_occl=block_size ** 2 / 2)

    img0_warped = list_img0_warped[0]
    dpt0_warped = list_dpt0_warped[0]
    dpt0_warped_fixed = dpt0_warped.copy()
    # print map_occl.shape, dpt0_warped_fixed.shape, dpt_res.shape
    # print map_occl
    dpt0_warped_fixed[map_occl] = dpt_res[map_occl]

    return dpt0_warped_fixed