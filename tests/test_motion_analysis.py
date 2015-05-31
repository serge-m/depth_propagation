__author__ = 'Sergey Matyunin'

import unittest
import algo.fastdeepflow
import algo.motion_analysis.occlusion_detection as od
from time import time
import logging
import os
import numpy
import utils.logger_setup

path_dir_cur = os.path.dirname(__file__)
path_dir_input = os.path.join(path_dir_cur, 'data/input', '')
path_reference_dir = os.path.join(path_dir_cur, 'data/reference', '')


class CoveredTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger()
        utils.logger_setup.setup_logging(default_path='logger_config.json', default_level=logging.DEBUG)

        flow_bwd = algo.fastdeepflow.read_flow(os.path.join(path_dir_input, 'motion/covered/flow_bwd.flo'))

        self.flow_bwd_check = map(lambda comp:comp.copy(), flow_bwd)
        self.flow_bwd_check[0][0:20,0:20] = -100
        self.flow_bwd_check[1][0:20,0:20] = -100


class TestFastVersion(CoveredTest):
    def test(self):
        start = time()
        covered = od.get_covered(*self.flow_bwd_check)
        self.logger.info("Time initial call {}".format(time()-start))

        start = time()
        covered2 = od.get_covered(*self.flow_bwd_check)
        time_covered = time()-start
        self.logger.info("time get_covered {}".format(time_covered))

        start = time()
        covered_reference = od.get_covered_reference(*self.flow_bwd_check)
        time_reference = time()-start
        self.logger.info("time get_covered_reference {}".format(time_reference))

        covered = numpy.array(covered)
        covered2 = numpy.array(covered2)
        covered_reference = numpy.array(covered_reference)

        self.assertTrue(numpy.array_equal(covered, covered2))
        self.assertTrue(numpy.allclose(covered, covered_reference, atol=1e-6))
        self.assertTrue(time_covered < 0.01 * time_reference,
                        "Speed of get_covered must be much better than speed of get_covered_reference")


class TestReference(CoveredTest):
    def test(self):
        covered_reference = od.get_covered_reference(*self.flow_bwd_check)

        loaded = numpy.load(os.path.join(path_reference_dir, "motion/covered/covered_reference.npz"))
        self.assertTrue(self, numpy.array_equal(loaded['covered_ref'], covered_reference))


if __name__ == '__main__':
    unittest.main()
