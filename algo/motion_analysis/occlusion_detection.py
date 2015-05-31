__author__ = 's'

import math

import numpy


def _update_covered(covered, i, j, coeff):
    if 0 <= i < covered.shape[0] and 0 <= j < covered.shape[1]:
        covered[i, j] += coeff


def get_covered_reference(u, v):
    if u.shape != v.shape:
        raise Exception("x and y components of the flow must have the same size")

    covered = numpy.zeros(shape=u.shape, dtype=numpy.float32)
    h, w = u.shape

    for i in xrange(h):
        for j in xrange(w):
            mx, my = u[i, j], v[i, j]
            mxf, myf = math.floor(mx), math.floor(my)
            cx, cy = mx-mxf, my-myf

            y_dst = i+int(myf)
            x_dst = j+int(mxf)
            _update_covered(covered, y_dst,   x_dst  , (1.-cy)*(1.-cx))
            _update_covered(covered, y_dst+1, x_dst  , (   cy)*(1.-cx))
            _update_covered(covered, y_dst  , x_dst+1, (1.-cy)*(   cx))
            _update_covered(covered, y_dst+1, x_dst+1, (   cy)*(   cx))

    return covered

