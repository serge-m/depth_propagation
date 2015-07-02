
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
            logger.debug("list_dist {}".format(list_dist))
            idx_best = numpy.argmin(list_dist)

            get_patch(dpt_res)[:, :] = get_patch(list_dpt0_warped[idx_best])
            if return_intermediate:
                get_patch(map_choice)[:, :] = idx_best
        #     distances = map(lambda warped_block: )
        #     break
        # break

    return dpt_res, dict(map_choice=map_choice)


# In[ ]:




# In[11]:

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
    
    plt.figure(figsize=(15,7), dpi=300)
    plt.imshow(img, **kwargs_imshow)
    if flow:
        vh.plot_optical_flow(*flow, stepu = 8, stepv = 8)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path)
    plt.close()

def generate_debug(s_img0, s_img1, s_dpt0, s_dpt1, name):
    s_flow_fwd = algo.fastdeepflow.calc_flow(s_img1, s_img0)
    s_flow_bwd = algo.fastdeepflow.calc_flow(s_img0, s_img1)
    
    s_dpt0_warped = algo.fastdeepflow.warp_image(s_dpt0, *s_flow_fwd)
    
    s_dpt0_warped_fixed, interm = op.compensate_and_fix_occlusions(s_img0, s_img1, s_flow_fwd, s_flow_bwd, s_dpt0, 
                                                           th_occl=0.9, 
                                                           return_intermediate=True, 
                                                           func_block_histo_choose=block_histo_choose1,
                                                                  )
    
    
    print interm.keys()
    map_choise = interm['data_block_choose']['map_choice']
    choice_flow = get_flow_choice(interm)
    
    def sf(*args, **kwargs):
        return sf_default("dump/"+name+"/", *args, **kwargs)
    
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




# In[12]:

path_dir_cur = "./tests/"
path_dir_input = os.path.join(path_dir_cur, 'data/input', '')
path_result_dir = os.path.join(path_dir_cur, 'data/results/dp_color_mapping', '')
path_reference_dir = os.path.join(path_dir_cur, 'data/reference', '')

def load_data(name,
              idx_frame_start = 1,
              list_idxs_to_load = [],
              slice_dim0=slice(None), slice_dim1=slice(None),
              ):

    templ_path_frm = os.path.join(path_dir_input, 'sintel/final/'    , name, 'frame_{:04d}.png')
    templ_path_dpt = os.path.join(path_dir_input, 'sintel/depth_viz/', name, 'frame_{:04d}.png')

    def load_and_crop(path):
        img = utils.image.imread(path)
        img = img[slice_dim0,slice_dim1,]
        return img

    img = [load_and_crop(templ_path_frm.format(idx_frame_start+i)) for i in list_idxs_to_load]
    dpt = [cv2.cvtColor(load_and_crop(templ_path_dpt.format(idx_frame_start+i)), cv2.COLOR_RGB2GRAY) for i in list_idxs_to_load]

    
    return img, dpt


# In[13]:

# rs = numpy.random.RandomState(0)
# a = rs.randint(0, 10, (10,20))
# sl = slice(None)
# print a
# print  a[sl]==a


# In[14]:

import tests.test_dp_color_mapping

# idx = 3
# img, dpt = tests.test_dp_color_mapping.load_data(list_idx_frame_as_start=[0,9])
# img0, img1 = img[idx], img[idx+1]
# dpt0, dpt1 = dpt_gt[idx], dpt_gt[idx+1]
# generate_debug(img0, img1, dpt0, dpt1, "sintel_crop")
# print "img1.shape", img1.shape

# generate_debug(img1, img0, dpt1, dpt0, "sintel_crop_reverse")


# In[ ]:

name = "alley_2"
slice_dim0=slice(150,350)
slice_dim1=slice(250,400)
list_idxs_to_load = [0,1]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[ ]:

name = "cave_2"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [0,1]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[10]:

name = "cave_2"
slice_dim0=slice(None)
slice_dim1=slice(None)
list_idxs_to_load = [8,9]
name_to_save = "{}_{}_{}_{}".format(name, slice_dim0, slice_dim1, list_idxs_to_load)
img, dpt = load_data(name, 1, list_idxs_to_load=list_idxs_to_load, slice_dim0=slice_dim0, slice_dim1=slice_dim1)

generate_debug(img[0], img[1], dpt[0], dpt[1], name_to_save)
generate_debug(img[1], img[0], dpt[1], dpt[0], name_to_save +"_rev")


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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
# TODO:
# 1) FG - short MV, BG - long MV. In backward direction FG also has short MV.
# So MV filtering cannot get long vectors to fix errors.
# ~/work/of/py/utils/dump/cave_2_slice(None, None, None)_slice(None, None, None)_[0, 1] - nead the head
# The same problem can be if FG BWD has long MV, and BG has short MV
# 2) too wide occlusion areas are bad also because they prevent from good matching thin occlusions with shifts from
# candidate set




