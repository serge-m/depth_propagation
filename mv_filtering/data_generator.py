import matplotlib.pyplot as plt
# plt.rcParams['image.cmap'] = 'gray'
# plt.rcParams['figure.figsize'] = (2000,2000)
# plt.rcParams['figure.dpi'] = 400
plt.rcParams['image.interpolation'] = 'none'


# In[2]:

import numpy
import os
import utils.logger_setup
import logging
utils.logger_setup.setup_logging(default_path='../logger_config.json', default_level=logging.DEBUG)
logger = logging.getLogger()
logger.info("started")


# In[3]:

import utils.image


path_img_dir = '../tests/data/input/artificial/50x25/templates/'
bg = utils.image.imread(os.path.join(path_img_dir, 'bg_frm.png'))
fg = utils.image.imread(os.path.join(path_img_dir, 'fg_frm.png'))


plt.imshow(bg)


# In[29]:

import cv2
img = [None,] * 2
# fg_s = fg[:,:10]
# coeff = 4
coeff_fg = 2
bg_b = cv2.resize(bg, dsize=(256, 128))
fg_b = cv2.resize(fg, dsize=(40, 24))
print fg_b.shape


# In[38]:

pos0 = ((numpy.array(bg_b.shape) - numpy.array(fg_b.shape) ) / 2)[:2][::-1]
shift = (48, 30)

path_base = './'
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
        plt.figure()
        plt.imshow(img[1])
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        utils.image.imwrite(os.path.join(path_dir, 'img0.png'), img[0])
        utils.image.imwrite(os.path.join(path_dir, 'img1.png'), img[1])
        

