__author__ = 's'
import multiprocessing
import logging

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
        """
        Calculate optical flow between adjacent frames of input image sequence

        :param images: input sequence of images
        :return: list of forward optical flows. First element of the sequence is None. result[i] is OF between
        image[i-1] and image[i]
        """
        logger = logging.getLogger(__name__)
        logger.debug("Creating pool of tasks")
        inputs = [(images[i], images[i-1], i) for i in range(1, len(images))]

        pool = multiprocessing.Pool(processes=self.pool_size)
        logger.debug("Running")
        pool_outputs = pool.map(do_calculation_tuple, inputs)
        pool.close() # no more tasks
        logger.debug("Closed")
        pool.join()  # wrap up current tasks
        logger.debug("Joined")
        return [None,] + pool_outputs


class MotionEstimationSequential(MotionEstimation):
    def __init__(self):
        if fastdeepflow is None:
            raise Exception("Failed to load optical flow library")

    def calc(self, images):
        flow_fwd = [None, ] + [do_calculation(images[i], images[i-1])
                               for i in range(1, len(images))]
        return flow_fwd
