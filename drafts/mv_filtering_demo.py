
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
# plt.rcParams['figure.figsize'] = %%!(2000,2000)
# plt.rcParams['figure.dpi'] = 400
plt.rcParams['image.interpolation'] = 'none'

import mpld3

# mpld3.enable_notebook()


# In[2]:

import numpy
import os
import algo
import utils.logger_setup
import logging
utils.logger_setup.setup_logging(default_path='./logger_config.json', default_level=logging.DEBUG)
logger = logging.getLogger()
logger.info("started")


# In[3]:

import utils.image
import utils.visualization_helpers as vh
import cv2


# In[ ]:




# In[4]:

algo.fastdeepflow = reload(algo.fastdeepflow)


# In[5]:

img0 = utils.image.imread('mv_filtering/20_0/img0.png')
img1 = utils.image.imread('mv_filtering/20_0/img1.png')


img0 = utils.image.imread('mv_filtering/-48_0/img0.png')
img1 = utils.image.imread('mv_filtering/-48_0/img1.png')

params = algo.fastdeepflow.create_params("middlebury")
params.min_size = 10  


# In[6]:

dpt0_real = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
t1 = dpt0_real.reshape(dpt0_real.shape+(1,))
t2 = img0

dpt0 = numpy.dstack([t2,t1])
# plt.imshow(dpt0[:,:,0:3])
x = dpt0[:,:,0:3]
plt.imshow(img0)
print dpt0.shape, dpt0.min(), x.min()

numpy.array_equal(dpt0[:,:,0:3],img0)


# In[7]:

flow_fwd = algo.fastdeepflow.calc_flow(img1, img0, params)
flow_bwd = algo.fastdeepflow.calc_flow(img0, img1, params)


# In[8]:

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(img1)
vh.plot_optical_flow(*flow_fwd)

plt.subplot(1,2,2)
plt.imshow(img0)
vh.plot_optical_flow(*flow_bwd)


# In[9]:

import algo.motion_analysis.occlusion_processing
algo.motion_analysis.occlusion_processing = reload(algo.motion_analysis.occlusion_processing)
import algo.motion_analysis.occlusion_processing as op

dpt0_warped_fixed = op.compensate_and_fix_occlusions(img0, img1, flow_fwd, flow_bwd, dpt0, th_occl=0.9)
print dpt0_warped_fixed.shape
plt.imshow(dpt0_warped_fixed[:,:,3].astype('uint8'))


# In[10]:

import tests.test_dp_color_mapping

img, dpt, dpt_gt = tests.test_dp_color_mapping.load_data(list_idx_frame_as_start=[0,9])


# In[11]:

print dpt[0].shape
plt.imshow(dpt_gt[1])
plt.imshow(img[0])


# In[12]:

import algo.motion_analysis.occlusion_processing
algo.motion_analysis.occlusion_processing = reload(algo.motion_analysis.occlusion_processing)
import algo.motion_analysis.occlusion_processing as op

block_size=20

def block_histo_choose1(map_occl, img1, list_img0_warped, list_dpt0_warped, th_num_occl, return_intermediate=False):
    logger = logging.getLogger(__name__)
    dpt_res = numpy.zeros_like(list_dpt0_warped[0])

    h, w, _ = img1.shape
    
    map_choice = numpy.zeros(shape=(h,w), dtype=numpy.int32)

    for h_start in xrange(0*block_size, h, block_size):
        for w_start in xrange(0*block_size, w, block_size):
            def get_patch(img):
                return img[h_start:h_start+block_size, w_start:w_start+block_size]

            img1_block = get_patch(img1)

            list_img0_warped_block = map(get_patch, list_img0_warped)
            map_occl_block = get_patch(map_occl)

            cnt_occl = numpy.count_nonzero(map_occl_block)
            if cnt_occl == 0:
                get_patch(dpt_res)[:, :] = 0
                continue
            
            if cnt_occl < th_num_occl:
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
            if return_intermediate:
                get_patch(map_choice)[:, :] = idx_best
        #     distances = map(lambda warped_block: )
        #     break
        # break

    return dpt_res, dict(map_choice=map_choice)


# In[13]:

s_img0 = img[3]
s_img1 = img[4]
s_dpt0 = dpt_gt[3]
s_dpt1 = dpt_gt[4]
s_flow_fwd = algo.fastdeepflow.calc_flow(s_img1, s_img0)
s_flow_bwd = algo.fastdeepflow.calc_flow(s_img0, s_img1)
s_dpt0_warped_fixed, interm = op.compensate_and_fix_occlusions(s_img0, s_img1, s_flow_fwd, s_flow_bwd, s_dpt0, 
                                                       th_occl=0.9, 
                                                       return_intermediate=True, 
                                                       func_block_histo_choose=block_histo_choose1)
s_dpt0_warped = algo.fastdeepflow.warp_image(s_dpt0, *s_flow_fwd)


# In[14]:

interm.keys(), interm['data_block_choose']['map_choice']


# In[15]:

# list_flow = interm['interm_dpt_candidates']['list_flow']
# numpy.array(list_flow).shape
# list_flow_u = numpy.array(list_flow)[:,0,:,:]
# list_flow_v = numpy.array(list_flow)[:,1,:,:]


# # list_flow_u[map_choise].shape



# choice_u, choice_v = list_flow_u[map_choise[iy, ix], iy, ix], list_flow_v[map_choise[iy, ix], iy, ix]
# du, dv = choice_u-list_flow[0][0], choice_v-list_flow[0][1]

# fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
# ax.imshow(map_choise, vmin=0, vmax=4)
# # plt.imshow(du**2+dv**2)
# # vh.plot_optical_flow(choice_u, choice_v)
# vh.plot_optical_flow(*list_flow[1])
# # vh.plot_optical_flow(du, dv)


# In[16]:

def get_flow_choice(imterm):
    list_flow = interm['interm_dpt_candidates']['list_flow']
    map_choise = interm['data_block_choose']['map_choice']
    ny, nx = map_choise.shape
    ix, iy = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
    flow_choice = numpy.array(list_flow)[map_choise[iy, ix], :,  iy, ix]
    flow_choice = (flow_choice[0], flow_choice[1])
    return flow_choice

map_choise = interm['data_block_choose']['map_choice']
fig, ax = plt.subplots(1,2, subplot_kw=dict(axisbg='#EEEEEE'))
ax.flat[0].imshow(map_choise, vmin=0, vmax=4)
ax.flat[1].imshow(map_choise==3, vmin=0, vmax=4)


# In[17]:

# vh = reload(utils.visualization_helpers)
# fig, ax = plt.subplots(2, 2, figsize=(8, 8),sharex=True, sharey=True)
# fig.subplots_adjust(hspace=0.0)
# ax.flat[0].imshow(img[0])
# ax.flat[1].imshow(img[1])
# ax.flat[2].imshow(map_choise)
# vh.plot_optical_flow(*list_flow[1], axes=ax.flat[2])
# # plt.savefig("123123.png")


# In[18]:

def plot1(img):
    
    ax.flat[plot1.count].imshow(img)
    plot1.count += 1

plot1.count = 0   


# In[19]:

# fig, ax = plt.subplots(6, 2 , figsize=(10, 20),sharex=True, sharey=True)
# list_flow = interm['interm_dpt_candidates']['list_flow']


# xx = [c.copy() for c in s_flow_bwd]
# xx[0][100:110,100:110] = -100
# xx[1][100:110,100:110] = -100

# ax.flat[0].imshow(s_img0)
# vh.plot_optical_flow(*s_flow_bwd, stepu=2, axes=ax.flat[0])

# ax.flat[1].imshow(s_img1)
# vh.plot_optical_flow(*s_flow_fwd, stepu=2, axes=ax.flat[1])

# ax.flat[2].imshow(s_dpt0)
# ax.flat[3].imshow(s_dpt1)


# start_idx_vis = 4
# for idx in range(5):
#     ax_c = ax.flat[start_idx_vis+idx]
#     ax_c.imshow(interm['list_img0_warped'][idx].astype('uint8'))
#     vh.plot_optical_flow(*list_flow[idx], stepu=2, axes=ax_c)

# ax.flat[9].imshow(utils.image.abs_diff(s_dpt0_warped, s_dpt0_warped_fixed))

# choice_flow = get_flow_choice(interm)
# vh.plot_optical_flow(*choice_flow, axes=ax.flat[9])
# ax.flat[10].imshow(interm['map_occl'])
# ax.flat[11].imshow(interm['list_dpt0_warped'][1])



# In[20]:

#fig, ax = plt.subplots(6, 2 , figsize=(10, 20),sharex=True, sharey=True)
list_flow = interm['interm_dpt_candidates']['list_flow']

def sf(path, img, flow=None):
    assert(path is not None)
    plt.figure(figsize=(20,20), dpi=600)
    plt.imshow(img)
    if flow:
        vh.plot_optical_flow(*flow)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path)
    plt.close()
    
sf("sintel_crop/s_img0_flow_bwd.png", s_img0, s_flow_bwd)
sf("sintel_crop/s_img0.png", s_img0)
sf("sintel_crop/s_img1_flow_fwd.png", s_img1, s_flow_fwd)
sf("sintel_crop/s_img1.png", s_img1)

sf("sintel_crop/s_dpt0.png", s_dpt0)
sf("sintel_crop/s_dpt1.png", s_dpt1)

start_idx_vis = 4
for idx in range(5):
    sf('sintel_crop/img0_warped_{}_flow.png'.format(idx), 
       interm['list_img0_warped'][idx].astype('uint8'),
       list_flow[idx])

# ax.flat[9].imshow(utils.image.abs_diff(s_dpt0_warped, s_dpt0_warped_fixed))

# choice_flow = get_flow_choice(interm)
# vh.plot_optical_flow(*choice_flow, axes=ax.flat[9])
# ax.flat[10].imshow(interm['map_occl'])
# ax.flat[11].imshow(interm['list_dpt0_warped'][1])


# In[21]:

utils.flow.flow_equal(s_dpt0_warped, )


# In[ ]:

123


# In[ ]:

# from PyQt4 import QtGui, QtCore

# class MyQtWindow(QtGui.QMainWindow):
#     # [...] your Qt window code
#     pass

# # app = QtGui.QApplication(["asdasd"])
# window = MyQtWindow()
# # window.show
# # app.exec_()


# In[ ]:

123


# In[ ]:




# In[ ]:




# In[20]:

123123


# In[ ]:




# In[ ]:



