__author__ = 's'
import multiprocessing
import logging
import os
import shelve
import hashlib
import numpy

from .motion_estimation import MotionEstimation


try:
    import fastdeepflow
except ImportError:
    raise Exception("Failed to load fastdeepflow module. Download from https://github.com/serge-m/deepflow")
    fastdeepflow = None


def do_calculation_tuple(tuple_parameters):
    logger = logging.getLogger()
    image_cur, image_ref, idx = tuple_parameters
    logger.info("Calculating optical flow for frame {}".format(idx))
    flow = do_calculation(image_cur, image_ref)
    logger.info("Calculated for frame {}".format(idx))
    return flow


def do_calculation(image_cur, image_ref):
    return fastdeepflow.calc_flow(image_cur, image_ref)


class MotionEstimationParallel(MotionEstimation):
    """
    Multiprocessing computation of optical flow for image sequence
    """
    def __init__(self, pool_size=8):
        """

        :param pool_size: numper of parallel processes used for optical flow computation
        :return:
        """
        if fastdeepflow is None:
            raise Exception("Failed to load optical flow library")

        self.pool_size = pool_size

    def calc(self, images):
        return [None,] + self.calc_pairs([(images[i], images[i-1]) for i in range(1, len(images))])

    def calc_pairs(self, pairs_images):
        """
        Calculate optical flow between adjacent frames of input image sequence

        :param images: input sequence of images
        :return: list of forward optical flows. First element of the sequence is None. result[i] is OF between
        image[i-1] and image[i]
        """
        logger = logging.getLogger(__name__)
        logger.debug("Creating pool of tasks")
        inputs = [(image_cur, image_ref, i) for i, (image_cur, image_ref) in enumerate(pairs_images)]

        pool = multiprocessing.Pool(processes=self.pool_size)
        logger.debug("Running")
        pool_outputs = pool.map(do_calculation_tuple, inputs)
        pool.close() # no more tasks
        logger.debug("Closed")
        pool.join()  # wrap up current tasks
        logger.debug("Joined")
        return pool_outputs


class MotionEstimationSequential(MotionEstimation):
    def __init__(self):
        if fastdeepflow is None:
            raise Exception("Failed to load optical flow library")

    def calc(self, images):
        flow_fwd = [None, ] + self.calc_pairs([(images[i], images[i-1])
                                               for i in range(1, len(images))])
        return flow_fwd

    def calc_pairs(self, pairs_images):
        return [do_calculation(image_cur, image_ref)
                for (image_cur, image_ref) in pairs_images]


class MotionEstimationParallelCached(MotionEstimationParallel):
    def __init__(self, path_cahce="cache.dat", **args_parallel):
        MotionEstimationParallel.__init__(self, **args_parallel)
        self.cache = shelve.open(path_cahce)

    def calc_pairs(self, pairs_images):
        logger = logging.getLogger(__name__)

        logger.debug("Generating hashes")
        list_hashes = [hash_of_img_pair(image_cur, image_ref) for (image_cur, image_ref) in pairs_images]
        list_idx_to_process = [i for i in range(len(pairs_images)) if list_hashes[i] not in self.cache]
        logger.debug("list_idx_to_process {}".format(list_idx_to_process))

        logger.debug("Hashes to process:")
        for idx in list_idx_to_process:
            logger.debug("{} {}".format(idx, list_hashes[idx]))


        pairs_images_to_process = [pairs_images[i] for i in list_idx_to_process]
        assert(len(pairs_images_to_process) == len(list_idx_to_process))

        processed = MotionEstimationParallel.calc_pairs(self, pairs_images_to_process)
        if len(processed) != len(pairs_images_to_process):
            raise Exception("Invalid flow")

        for idx, flow in zip(list_idx_to_process, processed):
            self.cache[list_hashes[idx]] = flow
            self.cache.sync()
            logger.debug("flow for hash {} saved".format(list_hashes[idx]))

        return [self.cache[hash_cur] for hash_cur in list_hashes]


def hash_of_img_pair(image_cur, image_ref):
    h0 = hashlib.sha1(numpy.ascontiguousarray(image_cur)).hexdigest()
    h1 = hashlib.sha1(numpy.ascontiguousarray(image_ref)).hexdigest()
    hash = h0 + "_" + h1
    return hash
