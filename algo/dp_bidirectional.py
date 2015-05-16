__author__ = 'Sergey Matyunin'
from depth_propagation_base import DepthPropagation
import numpy


class DepthPropagationBidir(DepthPropagation):
    def __init__(self, img, dpt, list_dp_classes):
        super(DepthPropagationBidir, self).__init__(img, dpt)
        self.list_dp_classes = list_dp_classes
        self.list_dp = [DPClass(img, dpt) for DPClass in self.list_dp_classes]

    def preprocess(self):
        for dp in self.list_dp:
            dp.preprocess()

        self.res = [numpy.average(list_res_cur, axis=0) for list_res_cur in zip(*self.list_dp)]



