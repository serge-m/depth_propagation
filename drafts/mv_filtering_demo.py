
# coding: utf-8

# In[1]:

# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (20,20)
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


# In[4]:

utils.image = reload(utils.image)


# In[5]:

import algo.motion_analysis.occlusion_processing
algo.motion_analysis.occlusion_processing = reload(algo.motion_analysis.occlusion_processing)
import algo.motion_analysis.occlusion_processing as op

block_size=20

def block_histo_choose1(map_occl, img1, list_img0_warped, list_dpt0_warped, th_num_occl, return_intermediate=False):
    img = 0
#     print map_occl.shape, [img.shape for img in list_img0_warped]
    
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
#                 print map_occl_block.shape, [img.shape for img in imgs]
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
#             list_dist = map(lambda block: utils.image.abs_diff(block, img1_block).sum(), list_img0_warped_block)
#             logger.debug("list_dist {}".format(list_dist))
            idx_best = numpy.argmin(list_dist)

            get_patch(dpt_res)[:, :] = get_patch(list_dpt0_warped[idx_best])
            if return_intermediate:
                get_patch(map_choice)[:, :] = idx_best
        #     distances = map(lambda warped_block: )
        #     break
        # break

    return dpt_res, dict(map_choice=map_choice)


# In[ ]:




# In[6]:

def get_flow_choice(interm):
    list_flow = interm['interm_dpt_candidates']['list_flow']
    map_choise = interm['data_block_choose']['map_choice']
    ny, nx = map_choise.shape
    ix, iy = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
    flow_choice = numpy.array(list_flow)[map_choise[iy, ix], :,  iy, ix]
#     print flow_choice.shape
    flow_choice = (flow_choice[:,:,0], flow_choice[:,:,1])
    return flow_choice


def sf_default(dir_base, path, img, flow=None, kwargs_imshow=dict()):
    assert(path is not None)
    path = os.path.join(dir_base, path)
    
    plt.figure(figsize=(15,7), dpi=400)
    plt.imshow(img, **kwargs_imshow)
    if flow:
        vh.plot_optical_flow(*flow, stepu = 8, stepv = 8)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path)
    plt.close()


# In[26]:

import algo.motion_estimation_sequence
def generate_debug(s_img0, s_img1, s_dpt0, s_dpt1, name):
#     s_flow_fwd = algo.fastdeepflow.calc_flow(s_img1, s_img0)
#     s_flow_bwd = algo.fastdeepflow.calc_flow(s_img0, s_img1)
    
    
    me = algo.motion_estimation_sequence.MotionEstimationParallelCached()
    s_flow_fwd, s_flow_bwd = me.calc_pairs([(s_img1, s_img0), (s_img0, s_img1)])

    th_occl = 0.5
    
    s_dpt0_warped = algo.fastdeepflow.warp_image(s_dpt0, *s_flow_fwd)
    
    s_dpt0_warped_fixed, interm = op.compensate_and_fix_occlusions(s_img0, s_img1, s_flow_fwd, s_flow_bwd, s_dpt0, 
                                                           th_occl=th_occl, 
                                                           return_intermediate=True, 
                                                           func_block_histo_choose=block_histo_choose1,
                                                                  )
    
    
    print interm.keys()
    map_choise = interm['data_block_choose']['map_choice']
    choice_flow = get_flow_choice(interm)
    
    def sf(*args, **kwargs):
        return sf_default("dump_th_occl{}/".format(th_occl)+name+"/", *args, **kwargs)
    
    sf("img0_flow_bwd.png", s_img0, s_flow_bwd)
    sf("img0.png", s_img0)
    sf("img1_flow_fwd.png", s_img1, s_flow_fwd)
    sf("img1.png", s_img1)

    sf("dpt0.png", s_dpt0)
    sf("dpt1.png", s_dpt1)


    start_idx_vis = 4
    list_flow = interm['interm_dpt_candidates']['list_flow']
    for idx in range(5):
        sf('img0_warped/img0_warped_{}_flow.png'.format(idx), 
           interm['list_img0_warped'][idx].astype('uint8'),
           list_flow[idx])
        sf('img0_warped/img0_warped_{}.png'.format(idx), 
           interm['list_img0_warped'][idx].astype('uint8'))
        
        sf('dpt0_warped/dpt0_warped_{}_flow.png'.format(idx), 
           interm['list_dpt0_warped'][idx].astype('uint8'),
           list_flow[idx])
        sf('dpt0_warped/dpt0_warped_{}.png'.format(idx), 
           interm['list_dpt0_warped'][idx].astype('uint8'))

    sf("map_choice.png", map_choise, kwargs_imshow=dict(vmin=0, vmax=4))
    sf("map_choice_flow.png", map_choise, flow=choice_flow, kwargs_imshow=dict(vmin=0, vmax=4))

    sf("map_occl.png", interm['map_occl'])
    sf("occl.png", interm['occl'])

    sf("map_occl_flow_bwd.png", interm['map_occl'], flow=s_flow_bwd)
    sf("occl_flow_bwd.png", interm['occl'], flow=s_flow_bwd)

    sf("map_occl_flow_fwd.png", interm['map_occl'], flow=s_flow_fwd)
    sf("occl_flow_fwd.png", interm['occl'], flow=s_flow_fwd)
    
    sf("s_dpt0_warped.png", s_dpt0_warped)
    sf("s_dpt0_warped_flow_fwd.png", s_dpt0_warped, s_flow_fwd)
    sf("s_dpt0_warped_fixed.png", s_dpt0_warped_fixed)
    sf("s_dpt0_warped_fixed_choice_flow.png", s_dpt0_warped_fixed, choice_flow)


# In[ ]:




# In[17]:

path_dir_cur = "./tests/"
path_dir_input = os.path.join(path_dir_cur, 'data/input', '')
path_result_dir = os.path.join(path_dir_cur, 'data/results/dp_color_mapping', '')
path_reference_dir = os.path.join(path_dir_cur, 'data/reference', '')


# In[18]:

def load_data(name,
              idx_frame_start = 1,
              list_idxs_to_load = [],
              slice_dim0=slice(None), slice_dim1=slice(None),
              ):
    
    list_idx = map(lambda i: i+idx_frame_start, list_idxs_to_load)
    templ_path_frm = os.path.join(path_dir_input, 'sintel/final/'    , name, 'frame_{:04d}.png')
    templ_path_dpt = os.path.join(path_dir_input, 'sintel/depth_viz/', name, 'frame_{:04d}.png')
    
    def load(templ, list_idx):
        list_paths = map(lambda idx: templ.format(idx), list_idx)
        img = utils.image.load_image_sequence(list_paths)
        img = utils.image.crop_image_sequence(img, slice_dim0, slice_dim1)
        return img
        
    img = load(templ_path_frm, list_idx)
    dpt = load(templ_path_dpt, list_idx)
    dpt = map(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), dpt)
        
    return img, dpt


# In[10]:

# rs = numpy.random.RandomState(0)
# a = rs.randint(0, 10, (10,20))
# sl = slice(None)
# print a
# print  a[sl]==a


# In[11]:

import tests.test_dp_color_mapping

# idx = 3
# img, dpt = tests.test_dp_color_mapping.load_data(list_idx_frame_as_start=[0,9])
# img0, img1 = img[idx], img[idx+1]
# dpt0, dpt1 = dpt_gt[idx], dpt_gt[idx+1]
# generate_debug(img0, img1, dpt0, dpt1, "sintel_crop")
# print "img1.shape", img1.shape

# generate_debug(img1, img0, dpt1, dpt0, "sintel_crop_reverse")


# In[12]:

# name = "alley_2"
# slice_dim0=slice(150,350)
# slice_dim1=slice(250,400)
# list_idxs_to_load = [0,1]
# name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
# img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

# generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
# generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[13]:

# import algo.motion_estimation_sequence
# me = algo.motion_estimation_sequence.MotionEstimationParallelCached()
# s_flow_fwd1, s_flow_bwd1 = me.calc_pairs([(s_img1, s_img0), (s_img0, s_img1)])
# utils.flow.flow_equal(s_flow_fwd1, s_flow_fwd)


# In[275]:

name = "cave_2"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [0,1]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)

img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

s_img0, s_img1, s_dpt0, s_dpt1 = img[0], img[1], dpt[0], dpt[1]

s_flow_fwd = algo.fastdeepflow.calc_flow(s_img1, s_img0)
s_flow_bwd = algo.fastdeepflow.calc_flow(s_img0, s_img1)


# In[326]:

s_dpt0_warped = algo.fastdeepflow.warp_image(s_dpt0, *s_flow_fwd)

s_dpt0_warped_fixed, interm = op.compensate_and_fix_occlusions(s_img0, s_img1, s_flow_fwd, s_flow_bwd, s_dpt0, 
                                                       th_occl=0.5, 
                                                       return_intermediate=True, 
                                                       func_block_histo_choose=block_histo_choose1,
                                                              )


# generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
# generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[338]:

img = img_as_float(s_img0[::1, ::1])
# img = s_flow_bwd[0]
segments_slic = slic(img, n_segments=100, compactness=10, sigma=1)
segments_slic = slic(img, n_segments=200, compactness=10, sigma=2,)

props = skimage.measure.regionprops(segments_slic)
flow_for_accum_u, flow_for_accum_v = [c.copy() for c in s_flow_bwd]
for pr in props:
    coords = pr.coords[:,0], pr.coords[:,1]
#     lu = flow_for_accum_u[coords]
#     lv = flow_for_accum_v[coords]
#     lu = pr.label
    flow_for_accum_u[coords] = flow_for_accum_u[coords].mean()
    flow_for_accum_v[coords] = flow_for_accum_v[coords].mean()


# In[339]:

s_dpt0_warped_fixed1, interm1 = op.compensate_and_fix_occlusions(s_img0, s_img1, s_flow_fwd, 
                                                               (flow_for_accum_u, flow_for_accum_v), s_dpt0, 
                                                       th_occl=0.5, 
                                                       return_intermediate=True, 
                                                       func_block_histo_choose=block_histo_choose1,
                                                              )


# In[344]:

from algo.motion_analysis.occlusion_processing import  calc_covered
occl = calc_covered(flow_bwd= s_flow_bwd)
map_occl = occl < 0.9
    
    
            
s_dpt0_warped_fixed2, interm2= op.compensate_and_fix_occlusions2(s_img0, 
                                     s_img1, 
                                     s_flow_fwd, 
                                     (flow_for_accum_u, flow_for_accum_v), 
                                     s_dpt0, 
                                     map_occl, 
                                     return_intermediate=True)


# In[345]:

print utils.image.abs_diff(s_dpt0_warped, s_dpt1).sum(), utils.image.abs_diff(s_dpt0_warped_fixed, s_dpt1).sum()
print utils.image.abs_diff(s_dpt0_warped, s_dpt1).sum(), utils.image.abs_diff(s_dpt0_warped_fixed1, s_dpt1).sum()
print utils.image.abs_diff(s_dpt0_warped, s_dpt1).sum(), utils.image.abs_diff(s_dpt0_warped_fixed2, s_dpt1).sum()


# In[346]:

# plt.figure(), plt.imshow(utils.image.abs_diff(s_dpt0_warped_fixed, s_dpt1))
# plt.figure(), plt.imshow(utils.image.abs_diff(s_dpt0_warped_fixed1, s_dpt1))


# In[347]:

plt.figure(), plt.imshow(s_dpt1)
plt.figure(), plt.imshow(s_dpt0_warped)
plt.figure(), plt.imshow(s_dpt0_warped_fixed)
plt.figure(), plt.imshow(s_dpt0_warped_fixed1)
plt.figure(), plt.imshow(s_dpt0_warped_fixed2)


# In[18]:

get_ipython().magic(u'matplotlib inline')
choice_flow = get_flow_choice(interm)
plt.imshow(choice_flow[0])
plt.figure()
plt.imshow(s_flow_bwd[0])


# In[259]:

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# img = s_img0[::2, ::2] #img_as_float(s_flow_bwd[0][::2, ::2])
# img = (img - img.mean()) / numpy.std(img)
# img = numpy.maximum(img, -1)
# img = numpy.minimum(img, 1)
img = img_as_float(s_img0[::1, ::1])
# logger.debug("1")
# segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
logger.debug("1")
segments_slic = slic(img, n_segments=100, compactness=10, sigma=1)
logger.debug("1")
# segments_quick = quickshift(img, kernel_size=3, max_dist=100, ratio=0.5)
# logger.debug("1")


# In[260]:

# import skimage.measure
# # numpy.max(segments_slic)
# ii = numpy.array([[1,1,2,3,4], [1,1,2,3,1]])
# vv = numpy.array([[10,20,30,40,50], [10,20,30,40,50]])
# ss = numpy.array([0,0,0])
# pp = skimage.measure.regionprops(ii)
# pr = pp[0]
# # vv[pr.coords.T]
# # vv.at([0,1])
# vv[numpy.where(ii==1)], pr.coords.T[0], vv[pr.coords[:,0], pr.coords[:,1]]


# In[ ]:




# In[261]:

props = skimage.measure.regionprops(segments_slic)
flow_for_accum_u, flow_for_accum_v = [c.copy() for c in s_flow_bwd]
for pr in props:
    coords = pr.coords[:,0], pr.coords[:,1]
#     lu = flow_for_accum_u[coords]
#     lv = flow_for_accum_v[coords]
#     lu = pr.label
    flow_for_accum_u[coords] = flow_for_accum_u[coords].mean()
    flow_for_accum_v[coords] = flow_for_accum_v[coords].mean()


# In[262]:

plt.imshow(flow_for_accum_v)
segments_slic.shape


# In[265]:

s_dpt0_warped_fixed1, interm1 = op.compensate_and_fix_occlusions(s_img0, s_img1, s_flow_fwd, 
                                                               (flow_for_accum_u, flow_for_accum_v), s_dpt0, 
                                                       th_occl=0.9, 
                                                       return_intermediate=True, 
                                                       func_block_histo_choose=block_histo_choose1,
                                                              )


# In[267]:

#  plt.imshow(interm1['map_occl'])


# In[232]:

# %matplotlib inline
# choice_flow1 = get_flow_choice(interm1)
# plt.imshow(choice_flow1[0])
# plt.figure()
# plt.imshow(s_flow_bwd[0])


# In[233]:

# plt.figure(figsize=(20,20))
# plt.imshow(flow_for_accum_u)
# vh.plot_optical_flow(flow_for_accum_u, flow_for_accum_v, stepu = 24, stepv = 24)


# In[288]:

# plt.figure(figsize=(10,10))
# segments_slic = slic(s_flow_bwd[0].astype('float64'), n_segments=200, compactness=10, sigma=2,)
# plt.imshow(mark_boundaries(img, segments_slic))


# In[273]:

# plt.imshow(s_flow_bwd[0].astype('float64'),)
print utils.image.abs_diff(s_dpt1, s_dpt0_warped_fixed).sum()
print utils.image.abs_diff(s_dpt1, s_dpt0_warped_fixed1).sum()


# In[96]:

logger.debug("2")
numpy.histogram(s_flow_bwd[1], bins=50)
logger.debug("2")
#_ = plt.hist(s_flow_bwd[1].ravel())


# In[215]:

# # plt.figure(figsize=(20,20))
# fig, ax = plt.subplots(3, 1, figsize=(20,20))
# # fig.set_size_inches(8, 8, forward=True)
# # fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)

# ax[0].imshow(mark_boundaries(img, segments_fz))
# ax[0].set_title("Felzenszwalbs's method")
# ax[1].imshow(mark_boundaries(img, segments_slic))
# vh.plot_optical_flow(flow_bwd)
# ax[1].set_title("SLIC")
# ax[2].imshow(mark_boundaries(img, segments_quick))
# ax[2].set_title("Quickshift")
# for a in ax:
#     a.set_xticks(())
#     a.set_yticks(())
# plt.show()


# In[27]:

name = "cave_2"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [0,1]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[28]:

name = "cave_2"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [8,9]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[29]:

name = "bamboo_1"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [0,1]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")

name = "bamboo_1"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [8,9]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[30]:

name = "alley_1"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [0,1]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")

name = "alley_1"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [8,9]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[31]:

name = "alley_2"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [0,1]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")

name = "alley_2"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [8,9]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[71]:

slice_dim0


# In[39]:

plt.subplot(221)
plt.imshow(img[0])
plt.subplot(222)
plt.imshow(img[1])

dpt


# In[33]:

# vh.plot_optical_flow(*choice_flow)
# choice_flow


# In[32]:


    



# ax.flat[9].imshow(utils.image.abs_diff(s_dpt0_warped, s_dpt0_warped_fixed))

# 
# vh.plot_optical_flow(*choice_flow, axes=ax.flat[9])
# imshow()
# ax.flat[11].imshow(interm['list_dpt0_warped'][1])


# In[29]:

interm.keys()


# In[ ]:




# In[ ]:




# In[20]:

123123


# In[ ]:




# In[ ]:



