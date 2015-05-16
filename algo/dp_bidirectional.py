__author__ = 'Sergey Matyunin'
from depth_propagation_base import DepthPropagation
import numpy
import logging


class DepthPropagationBidir(DepthPropagation):
    def __init__(self, img, dpt, list_dp_classes, strategy='average'):
        super(DepthPropagationBidir, self).__init__(img, dpt)
        self.logger = logging.getLogger(__name__)
        self.list_dp_classes = list_dp_classes
        self.list_dp = [DPClass(img, dpt) for DPClass in self.list_dp_classes]
        self.strategy = strategy

    def preprocess(self):
        for dp in self.list_dp:
            dp.preprocess()

        if self.strategy == 'average':
            self.res = [numpy.average(list_res_cur, axis=0) for list_res_cur in zip(*self.list_dp)]
        elif self.strategy == 'wto':
            for idx_frame in range(self.length):
                self.logger.debug("idx_frame {}".format(idx_frame))
                list_res_cur = zip(*self.list_dp)[idx_frame]
                self.logger.debug("len list_res_cur {}".format(len(list_res_cur)))

                list_dp_dist_cur = [dp.get_propagation_distance(idx_frame=idx_frame) for dp in self.list_dp]
                list_dp_dist_cur = numpy.array(list_dp_dist_cur, dtype=numpy.float32)
                self.logger.debug("list_dp_dist_cur {}".format(list_dp_dist_cur))

                best = numpy.argmin(list_dp_dist_cur)

                self.res[idx_frame] = list_res_cur[best]

        else:
            raise Exception("Unsupported strategy")

