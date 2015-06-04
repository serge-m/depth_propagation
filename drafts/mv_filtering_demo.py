
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.rcParams['image.cmap'] = 'gray'
# plt.rcParams['figure.figsize'] = (2000,2000)
# plt.rcParams['figure.dpi'] = 400
plt.rcParams['image.interpolation'] = 'none'


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


# In[50]:

dpt0_real = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
t1 = dpt0_real.reshape(dpt0_real.shape+(1,))
t2 = img0

dpt0 = numpy.dstack([t2,t1])
# plt.imshow(dpt0[:,:,0:3])
x = dpt0[:,:,0:3]
plt.imshow(img0)
print dpt0.shape, dpt0.min(), x.min()

numpy.array_equal(dpt0[:,:,0:3],img0)


# In[51]:

flow_fwd = algo.fastdeepflow.calc_flow(img1, img0, params)
flow_bwd = algo.fastdeepflow.calc_flow(img0, img1, params)


# In[52]:

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(img1)
vh.plot_optical_flow(*flow_fwd)

plt.subplot(1,2,2)
plt.imshow(img0)
vh.plot_optical_flow(*flow_bwd)


# In[66]:

import algo.motion_analysis.occlusion_processing
algo.motion_analysis.occlusion_processing = reload(algo.motion_analysis.occlusion_processing)
import algo.motion_analysis.occlusion_processing as op

dpt0_warped_fixed = op.compensate_and_fix_occlusions(img0, img1, flow_fwd, flow_bwd, dpt0, th_occl=0.9)
print dpt0_warped_fixed.shape
plt.imshow(dpt0_warped_fixed[:,:,3].astype('uint8'))


# In[63]:

import tests.test_dp_color_mapping

img, dpt, dpt_gt = tests.test_dp_color_mapping.load_data(list_idx_frame_as_start=[0,9])


# In[64]:

print dpt[0].shape
plt.imshow(dpt_gt[1])
plt.imshow(img[0])


# In[84]:

s_img0 = img[3]
s_img1 = img[4]
s_dpt0 = dpt_gt[3]
s_flow_fwd = algo.fastdeepflow.calc_flow(s_img1, s_img0)
s_flow_bwd = algo.fastdeepflow.calc_flow(s_img0, s_img1)
s_dpt0_warped_fixed = op.compensate_and_fix_occlusions(s_img0, s_img1, s_flow_fwd, s_flow_bwd, s_dpt0, th_occl=0.9)
s_dpt0_warped = algo.fastdeepflow.warp_image(s_dpt0, *s_flow_fwd)


# In[86]:

print dpt0_warped_fixed.shape
plt.figure()
plt.imshow(dpt_gt[4])
plt.figure()
plt.imshow(s_dpt0_warped_fixed[:,:].astype('uint8'))
plt.figure()
plt.imshow(s_dpt0_warped)
plt.figure()
plt.imshow(utils.image.abs_diff(s_dpt0_warped, s_dpt0_warped_fixed))


# In[ ]:




# In[ ]:



