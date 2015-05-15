__author__ = 'Sergey Matyunin'
import numpy
import logging


def flow_equal(flow1, flow2, verbose=False):
    """
    Compares two optical flows
    :param flow1:
    :param flow2:
    :param verbose:
    :return:
    """
    logger = logging.getLogger()
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if len(flow1) != len(flow2):
        logger.info("length different")
        return False

    for idx, f1, f2 in zip(range(len(flow1)), flow1, flow2):
        if f1 is None:
            if f2 is not None:
                logger.info("idx {}. None differs".format(idx))
                return False
            continue

        if not numpy.array_equal(numpy.array(f1), numpy.array(f2)):
            logger.info("idx {}. Non equal".format(idx))
            return False

    logger.info("Equal")
    return True