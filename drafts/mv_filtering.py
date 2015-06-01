
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
# plt.rcParams['image.cmap'] = 'gray'
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


# In[4]:

path_img_dir = 'tests/data/input/artificial/50x25/templates/'
bg = utils.image.imread(os.path.join(path_img_dir, 'bg_frm.png'))
fg = utils.image.imread(os.path.join(path_img_dir, 'fg_frm.png'))


# In[5]:

plt.imshow(bg)


# In[6]:

import cv2
img = [None,] * 2
# fg_s = fg[:,:10]
# coeff = 4
coeff_fg = 2
bg_b = cv2.resize(bg, dsize=(256, 128))
fg_b = cv2.resize(fg, dsize=(40, 24))
print fg_b.shape


# In[7]:

pos0 = ((numpy.array(bg_b.shape) - numpy.array(fg_b.shape) ) / 2)[:2][::-1]
shift = (48, 30)

path_base = 'mv_filtering/'
for shift in ((48, 30), (20, 10)):
    deltas = numpy.array([[1,0], [0, 1], [-1,0], [0,-1]])*shift
    for dx, dy in deltas:
       
        pos = pos0 + (dx,dy)
        path_dir = path_base + "/{}_{}/".format(dx, dy)
        print pos0, pos
        # fg_b = fg
        img[0] = utils.image.overlay(fg_b, bg_b, *pos0)
        img[1] = utils.image.overlay(fg_b, bg_b, *pos)
    #     img[1] = utils.image.overlay(fg_b, img[1], *pos0)
    #     plt.imshow(img[0])
#         plt.figure()
#         plt.imshow(img[1])
#         if not os.path.exists(path_dir):
#             os.makedirs(path_dir)
#         utils.image.imwrite(os.path.join(path_dir, 'img0.png'), img[0])
#         utils.image.imwrite(os.path.join(path_dir, 'img1.png'), img[1])
        


# In[8]:

import utils.visualization_helpers as vh
import algo.fastdeepflow

img0 = utils.image.imread('mv_filtering/48_0/img0.png')
img1 = utils.image.imread('mv_filtering/48_0/img1.png')
    
def show1(path_flow):
    u, v = algo.fastdeepflow.read_flow(path_flow)

    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(img0)
    vh.plot_optical_flow(u, v)
    w = algo.fastdeepflow.warp_image(img1, u, v)
    plt.subplot(1,2,2)
    plt.imshow(w.astype('uint8'))
show1('48.flo')
show1('48_midl.flo')


# In[9]:

algo.fastdeepflow = reload(algo.fastdeepflow)
import ctypes
def show2():
    img0 = utils.image.imread('mv_filtering/48_0/img0.png')
    img1 = utils.image.imread('mv_filtering/48_0/img1.png')
    params = algo.fastdeepflow.create_params("middlebury")
    params.min_size = 10
    u, v = algo.fastdeepflow.calc_flow(img0, img1, params)
    u[pos0[1]:pos0[1]+24, pos0[0]:pos0[0]+40] = 0

    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.imshow(img0)
    vh.plot_optical_flow(u, v)
    [numpy.abs(comp).max() for comp in [u,v]]
    w = algo.fastdeepflow.warp_image(img1, u, v)
    plt.subplot(2,1,2)
    plt.imshow(w.astype('uint8'))
show2()
pos0


# In[10]:

# params = algo.fastdeepflow.optical_flow_params_t()
# # algo.fastdeepflow.lib.optical_flow_params_middlebury(ctypes.byref(params))
# params.min_size


# In[ ]:




# In[11]:

utils.flow.flow_equal(f1, (u,v)), numpy.array_equal(f2[0], u)


# In[ ]:

import utils.visualization_helpers as vh
import algo.fastdeepflow
def show1(path_flow):
    img0 = utils.image.imread('mv_filtering/20_0/img0.png')
    img1 = utils.image.imread('mv_filtering/20_0/img1.png')
    u, v = algo.fastdeepflow.read_flow(path_flow)

    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(img0)
    vh.plot_optical_flow(u, v)
    [numpy.abs(comp).max() for comp in [u,v]]
    w = algo.fastdeepflow.warp_image(img1, u, v)
    plt.subplot(1,2,2)
    plt.imshow(w.astype('uint8'))
show1('20_default_minsize10.flo')
show1('20_midl_minsize10.flo')


# In[ ]:

# Occlusions processing


# In[12]:

img0 = utils.image.imread('mv_filtering/20_0/img0.png')
img1 = utils.image.imread('mv_filtering/20_0/img1.png')
params = algo.fastdeepflow.create_params("middlebury")
params.min_size = 10  


# In[13]:

flow_fwd = algo.fastdeepflow.calc_flow(img1, img0, params)
flow_bwd = algo.fastdeepflow.calc_flow(img0, img1, params)


# In[14]:

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(img1)
vh.plot_optical_flow(*flow_fwd)

plt.subplot(1,2,2)
plt.imshow(img0)
vh.plot_optical_flow(*flow_bwd)


# In[15]:

import numpy
def get_covered_floor(u, v):
    if u.shape != v.shape:
        raise Exception("x and y components of the flow must have the same size")
    
    flow = numpy.dstack([v, u])
    print flow.shape
    covered = numpy.zeros(shape=u.shape, dtype=numpy.float32)
    h, w = u.shape
    
    def update_covered(i, j, coeff):
        if i >= 0 and i < h and j >= 0 and j < w:
            covered[i,j] += coeff
    
    for i in xrange(h):
        for j in xrange(w):
            mv = flow[i, j]
            mv_floor = numpy.floor(mv)
            coeff = mv - mv_floor
            
            update_covered(i+mv_floor[0], j+mv_floor[1], 1)
            
    return covered


# In[17]:

import math
int(0.99), int(-0.99), math.floor(-0.99)


# In[22]:

import numba
from time import time

@numba.jit
def _update_covered(covered, i, j, coeff):
    if 0 <= i < covered.shape[0] and 0 <= j < covered.shape[1]:
        covered[i, j] += coeff
        
@numba.jit
def get_covered(u, v):
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



flow_bwd_check = map(lambda comp:comp.copy(), flow_bwd)
flow_bwd_check[0][0:20,0:20] = -100
flow_bwd_check[1][0:20,0:20] = -100


# In[ ]:




# In[73]:

import algo.motion_analysis.occlusion_detection as od
start = time()
covered = od.get_covered(*flow_bwd_check)
logger.info("Time_get_covered {}".format(time()-start))
plt.imshow(covered, cmap=plt.cm.gray, vmax=1, vmin=0)

covered = numpy.clip(covered, 0, 1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
morph = cv2.morphologyEx(covered, cv2.MORPH_OPEN, kernel)
plt.figure()
plt.imshow(morph,  cmap=plt.cm.gray, vmax=1)
plt.figure()
plt.imshow(utils.image.abs_diff(morph, covered))


# In[101]:

t = 1-morph
block_size = 20
h_start = 2*block_size
w_start = 5*block_size
def get_patch(img):
    return img[h_start:h_start+block_size, w_start:w_start+block_size]
t[h_start:h_start+block_size, w_start:w_start+block_size] = 0.5

morph_block = get_patch(morph)
flow_bwd_block = map(get_patch, flow_bwd)
u1, v1 = flow_bwd_block
th_morph = 0.9
where_low_th = morph_block<th_morph

# u1[where_low_th] = numpy.median(u1[where_low_th])
# v1[where_low_th] = numpy.median(v1[where_low_th])

plt.imshow(t, cmap=plt.cm.gray)

plt.figure()
plt.imshow(morph_block)
vh.plot_optical_flow(*flow_bwd_block, stepu=2, stepv=2)

plt.figure()
plt.imshow(morph_block)
vh.plot_optical_flow(*flow_bwd_block, stepu=2, stepv=2)

numpy.median(u1s)


# In[322]:

# th_morph = 0.9
# block_size = 20

# def block_mv_median_for_occlusions(flow_bwd):
#     flow_bwd = map(lambda x:x.copy(), flow_bwd)
#     h, w = flow_bwd[0].shape
#     for h_start in xrange(0, h, block_size):
#         for w_start in xrange(0, w, block_size):
#             def get_patch(img):
#                 return img[h_start:h_start+block_size, w_start:w_start+block_size]

#             morph_block = get_patch(morph)
#             flow_bwd_block = map(get_patch, flow_bwd)
#             u1, v1 = flow_bwd_block

#             where_low_th = morph_block<th_morph

#             u1[where_low_th] = numpy.median(u1[where_low_th])
#             v1[where_low_th] = numpy.median(v1[where_low_th])

#     return flow_bwd

# start = time()
# flow_bwd_filt = block_mv_median_for_occlusions(flow_bwd)
# logger.info("block_mv_median_for_occlusions time {}".format(time()-start))

# # plt.imshow(morph, cmap=plt.cm.gray)

# list_coeffs = numpy.array([[1,1], [0.5,0.5], [-1,-1], [-0.5,-0.5]], dtype=numpy.float32)

# list_img0_warped = []
# list_flow_bwd_filt_coeff = []
# for coeff in list_coeffs:
#     flow_bwd_filt_coeff = coeff[0]*flow_bwd_filt[0], coeff[1]*flow_bwd_filt[1]
#     list_flow_bwd_filt_coeff.append(flow_bwd_filt_coeff)
#     list_img0_warped.append(algo.fastdeepflow.warp_image(img0, *flow_bwd_filt_coeff))
#     print coeff
# #     plt.figure()
# #     plt.imshow(list_img0_warped[-1].astype('uint8'))



# ############################
# list_dpt0_warped = list_img0_warped
# map_morph = morph<th_morph
# # def block_histo_choose(map_morph, img0, list_img0_warped, list_flow_bwd_filt_coeff, list_dpt0_warped):
# u_res, v_res = numpy.zeros_like(list_flow_bwd_filt_coeff[0][0]), numpy.zeros_like(list_flow_bwd_filt_coeff[0][1])

# dpt_res = numpy.zeros_like(list_dpt0_warped[0])

# h, w, _ = img0.shape
# #     def calc_hist(img0_block):
# #     #     plt.figure()
# #     #     plt.imshow(img0_block.astype('uint8'))
# #         #hist = cv2.calcHist(img0_block.astype('uint8'), [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
# #         hist = cv2.calcHist([img0_block.astype('uint8')], [0, 1, 2], None, [8,]*3, [0, 256]*3)
# #     #     hist = cv2.normalize(hist)
# #         hist = hist.flatten()
# #     #     print hist
# #         return hist

# for h_start in xrange(2*block_size, h, block_size):
#     for w_start in xrange(5*block_size, w, block_size):
#         def get_patch(img):
#             return img[h_start:h_start+block_size, w_start:w_start+block_size]

#         img0_block = get_patch(img0)

# #         print hist.shape, hist.dtype

#         list_img0_warped_block = map(get_patch, list_img0_warped)
#         map_morph_block = get_patch(map_morph)

#         plt.figure()
#         plt.imshow(map_morph_block)
#         plt.figure()
#         plt.imshow(img0_block)

#         plt.figure()
#         for idx, bl in enumerate(list_img0_warped_block):
#             plt.subplot(2,2, idx+1)
#             plt.imshow(bl.astype('uint8'))

#         def calc_hist(img0_block):
#             hist = cv2.calcHist([img0_block.astype('uint8')], [0, 1, 2], 
#                                 map_morph_block.astype('uint8'), [8,]*3, [0, 256]*3)
#             hist = cv2.normalize(hist)
# #                 hist = hist.flatten()
#             return hist

#         hist = calc_hist(img0_block)
#         list_hist = map(calc_hist, list_img0_warped_block)
#         #print "list_hist", len(list_hist), [hi.shape for hi in list_hist], list_hist

#         plt.figure()
#         plt.plot(hist.ravel(), )
#         plt.figure(figsize=(20,10))
#         for idx, hi in enumerate(list_hist):
#             plt.subplot(2,2, idx+1)
#             plt.plot(hi.ravel(), marker='o+xs*^'[idx])
# #                 print "d", numpy.abs(hist-hi).sum()
# #                 print list_hist[idx]
#         list_dist = map(lambda hist1: cv2.compareHist(hist1, hist, cv2.cv.CV_COMP_CHISQR), list_hist)
#         print "list_dist", list_dist
#         idx_best = numpy.argmin(list_dist)

#         get_patch(u_res)[:,:] = get_patch(list_flow_bwd_filt_coeff[idx_best][0])
#         get_patch(v_res)[:,:] = get_patch(list_flow_bwd_filt_coeff[idx_best][1])
#         get_patch(dpt_res)[:,:] = get_patch(list_dpt0_warped[idx_best])
# #             distances = map(lambda warped_block: )
#         break
#     break


# #     return u_res, v_res, dpt_res

# print img0.shape
# u_res, v_res, dpt_res = block_histo_choose(morph<th_morph, img1, list_img0_warped, list_flow_bwd_filt_coeff, list_img0_warped)
    
# # plt.figure()
# # plt.imshow(morph)
# # vh.plot_optical_flow(*flow_bwd, stepu=8, stepv=4)

# # plt.figure()
# # plt.imshow(morph)
# # vh.plot_optical_flow(*flow_bwd_filt, stepu=8, stepv=4)

# # numpy.median(u1s)

# # plt.figure()
# # plt.imshow(img1)
# # plt.figure()
# # plt.imshow(img0)
# plt.figure()
# img0_warped = algo.fastdeepflow.warp_image(img0, *flow_fwd).astype('uint8')
# plt.imshow(img0_warped)
# img0_warped_fixed = img0_warped.copy()
# img0_warped_fixed[morph<th_morph] = dpt_res[morph<th_morph]
# plt.figure()
# plt.imshow(dpt_res.astype('uint8'))
# plt.figure()
# plt.imshow(img0_warped_fixed.astype('uint8'))


# In[339]:

th_morph = 0.9
block_size = 20

def block_mv_median_for_occlusions(flow_bwd):
    flow_bwd = map(lambda x:x.copy(), flow_bwd)
    h, w = flow_bwd[0].shape
    for h_start in xrange(0, h, block_size):
        for w_start in xrange(0, w, block_size):
            def get_patch(img):
                return img[h_start:h_start+block_size, w_start:w_start+block_size]

            morph_block = get_patch(morph)
            flow_bwd_block = map(get_patch, flow_bwd)
            u1, v1 = flow_bwd_block

            where_low_th = morph_block<th_morph

            u1[where_low_th] = numpy.median(u1[where_low_th])
            v1[where_low_th] = numpy.median(v1[where_low_th])

    return flow_bwd

start = time()
flow_bwd_filt = block_mv_median_for_occlusions(flow_bwd)
logger.info("block_mv_median_for_occlusions time {}".format(time()-start))

# plt.imshow(morph, cmap=plt.cm.gray)

list_coeffs = numpy.array([[1,1], [0.5,0.5], [-1,-1], [-0.5,-0.5]], dtype=numpy.float32)

list_img0_warped = []
list_flow_bwd_filt_coeff = []
for coeff in list_coeffs:
    flow_bwd_filt_coeff = coeff[0]*flow_bwd_filt[0], coeff[1]*flow_bwd_filt[1]
    list_flow_bwd_filt_coeff.append(flow_bwd_filt_coeff)
    list_img0_warped.append(algo.fastdeepflow.warp_image(img0, *flow_bwd_filt_coeff))
    print coeff

############################
list_dpt0_warped = list_img0_warped
map_morph = morph<th_morph
def block_histo_choose(map_morph, img0, list_img0_warped, list_flow_bwd_filt_coeff, list_dpt0_warped):
    u_res, v_res = numpy.zeros_like(list_flow_bwd_filt_coeff[0][0]), numpy.zeros_like(list_flow_bwd_filt_coeff[0][1])

    dpt_res = numpy.zeros_like(list_dpt0_warped[0])

    h, w, _ = img0.shape

    for h_start in xrange(0*block_size, h, block_size):
        for w_start in xrange(0*block_size, w, block_size):
            def get_patch(img):
                return img[h_start:h_start+block_size, w_start:w_start+block_size]

            img0_block = get_patch(img0)

            list_img0_warped_block = map(get_patch, list_img0_warped)
            map_morph_block = get_patch(map_morph)

            if 0:
                plt.figure()
                plt.imshow(map_morph_block)
                plt.figure()
                plt.imshow(img0_block)

                plt.figure()
                for idx, bl in enumerate(list_img0_warped_block):
                    plt.subplot(2,2, idx+1)
                    plt.imshow(bl.astype('uint8'))

            def calc_hist(img0_block):
                imgs = [img0_block[:,:,i].astype('uint8') for i in range(3)]
                hist = [cv2.calcHist([img], [0,],  map_morph_block.astype('uint8'), [8,], [0, 256], ) for img in imgs]
                hist = numpy.vstack(hist)
            #     print hist
                hist = cv2.normalize(hist)
                return hist

            hist = calc_hist(img0_block)
            list_hist = map(calc_hist, list_img0_warped_block)
            #print "list_hist", len(list_hist), [hi.shape for hi in list_hist], list_hist

            if 0:
                plt.figure()
                plt.plot(hist.ravel(), )
    #             plt.figure(figsize=(20,10))
                for idx, hi in enumerate(list_hist):
    #                 plt.subplot(2,2, idx+1)
                    plt.plot(hi.ravel(), '+x.*'[idx])
        #                 print "d", numpy.abs(hist-hi).sum()
        #                 print list_hist[idx]
            list_dist = map(lambda hist1: cv2.compareHist(hist1, hist, cv2.cv.CV_COMP_BHATTACHARYYA), list_hist)
            print "list_dist", list_dist
            idx_best = numpy.argmin(list_dist)

            get_patch(u_res)[:,:] = get_patch(list_flow_bwd_filt_coeff[idx_best][0])
            get_patch(v_res)[:,:] = get_patch(list_flow_bwd_filt_coeff[idx_best][1])
            get_patch(dpt_res)[:,:] = get_patch(list_dpt0_warped[idx_best])
    #             distances = map(lambda warped_block: )
#             break
#         break


    return u_res, v_res, dpt_res

print img0.shape
u_res, v_res, dpt_res = block_histo_choose(morph<th_morph, img1, list_img0_warped, list_flow_bwd_filt_coeff, list_img0_warped)


plt.figure()
img0_warped = algo.fastdeepflow.warp_image(img0, *flow_fwd).astype('uint8')
plt.imshow(img0_warped)
img0_warped_fixed = img0_warped.copy()
img0_warped_fixed[morph<th_morph] = dpt_res[morph<th_morph]
plt.figure()
plt.imshow(dpt_res.astype('uint8'))
plt.figure()
plt.imshow(img0_warped_fixed.astype('uint8'))


# In[261]:

cv2.compareHist(hi.ravel(), hist.ravel(), method=cv2.cv.CV_COMP_CHISQR )
# hist


# In[278]:

img0_block.shape


# In[311]:

def calc_hist_d(img0_block):
    hist = cv2.calcHist([img0_block.astype('uint8')], [0, 1, 2], 
                        map_morph_block.astype('uint8'), [16,]*3, [0, 256]*3, )
    hist = cv2.normalize(hist)
    return hist

def calc_hist_d2(img0_block):
    imgs = [img0_block[:,:,i].astype('uint8') for i in range(3)]
    hist = [cv2.calcHist([img], [0,],  map_morph_block.astype('uint8'), [16,], [0, 256], ) for img in imgs]
    hist = numpy.vstack(hist)
#     print hist
    hist = cv2.normalize(hist)
    return hist

hhh = calc_hist_d2(img0_block)
print hhh.shape
plt.plot(hhh)
# Q = hhh.cumsum()
# plt.imshow(hhh.sum(axis=0), vmin=0, vmax=1)
# plt.figure()
# Q.shape
# # plt.imshow(Q.sum(axis=0), vmin=0, vmax=1)
# numpy.cumsum(hhh)#.shape


# In[283]:

img0_block[:,:,0].shape


# In[340]:

for idx, bl in enumerate(list_img0_warped_block):
    plt.subplot(2,2, idx+1)
    plt.imshow(bl.astype('uint8'))
    
plt.figure()
plt.imshow(img0_block)
plt.figure()
t0 = calc_hist_d2(img0_block)
for idx, bl in enumerate(list_img0_warped_block):
    plt.subplot(2,2, idx+1)
    #cv2.calcHist(bl.astype('uint8'), [0,1,2], None, histSize=[8,]*3, ranges=[0, 256,]*3)
#     cv2.calcHist(bl.astype('uint8'), [0, 1, 2], None, [8,]*3, [0, 256]*3)
    t1 = calc_hist_d2(bl)
    plt.plot(t1)
    print cv2.compareHist(t0, t1, cv2.cv.CV_COMP_BHATTACHARYYA)
    
plt.figure()
plt.plot(t0)


# In[151]:

dpt_res.max()


# In[132]:

list_flow_bwd_filt_coeff[0][2].shape


# In[27]:

import algo.motion_analysis.occlusion_detection as od
# od = reload(algo.motion_analysis.occlusion_detection)


start = time()
covered_ref = od.get_covered_reference(*flow_bwd_check)
logger.info("od.get_covered_reference() {}".format(time()-start))
plt.imshow(covered_ref, cmap=plt.cm.gray, vmax=2, vmin=0)

plt.figure()

start = time()
covered_lib = od.get_covered(*flow_bwd_check)
logger.info("od.get_covered() {}".format(time()-start))
plt.imshow(covered_lib, cmap=plt.cm.gray, vmax=2, vmin=0)


diff = utils.image.abs_diff(covered, covered_ref)
print diff.max()
print numpy.where(diff!=0)[0]#@.shape, 256*128
# print list(numpy.where(diff!=0)[0])
diff[diff!=0] = 1
plt.imshow(diff)


# In[ ]:




# In[ ]:




# In[81]:

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(img1)
vh.plot_optical_flow(*flow_fwd)

plt.subplot(1,2,2)
plt.imshow(img0)
vh.plot_optical_flow(*flow_bwd)


# In[100]:

flow_occl = map(lambda x:-1.5 * x, flow_bwd)
plt.imshow(algo.fastdeepflow.warp_image(img1, *flow_occl).astype(numpy.uint8))
vh.plot_optical_flow(*flow_occl)


# In[87]:

plt.imshow(algo.fastdeepflow.warp_image(img0, *flow_fwd).astype(numpy.uint8))
vh.plot_optical_flow(*flow_fwd)


# In[ ]:



