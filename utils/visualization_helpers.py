__author__ = 'Sergey Matyunin'
import matplotlib.pyplot as plt
import numpy
import numpy as np

def plot_optical_flow(nu, nv, stepu=4, stepv=4, axes=plt):
    u = nu[::stepu, ::stepv]
    v = nv[::stepu, ::stepv]
    
    shape = nu.shape
    idx, idy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    idx, idy = idx[::stepu, ::stepv], idy[::stepu, ::stepv]

    axes.quiver(idx, idy, u, v, scale_units='xy', angles='xy', scale=1., color='g')


def save_optical_flow_vis(path_dst, u, v, stepu=4, stepv=4):
    # fig = plt.figure()
    # plt.imshow(img_background)
    plot_optical_flow(u, v, stepu, stepv)
    plt.xlim(-1, u.shape[1])
    plt.ylim(u.shape[0], -1, )
    plt.savefig(path_dst, bbox_inches='tight')
    # plt.close(fig)