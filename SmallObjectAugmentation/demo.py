import os
import random
from os.path import join
import aug
import Helpers as hp
from util import *

# ###########Pipeline##############
"""
1 准备数据集和yolo格式标签, 如果自己的数据集是voc或coco格式的，先转换成yolo格式，增强后在转回来
2 run crop_image.py  裁剪出目标并保存图片
3 run demo.py   随机将裁剪出目标图片贴到需要增强的数据集上，并且保存增强后的图片集和label文件
"""

class2id = {"plane": 0}

cl = 'plane'  # 在这里更改 你要转换的类别
times = 1  # 更改每次在 原图上 添加多少个小图像

base_dir = './'

cl_id = class2id[cl]
print('cl_id',cl_id)

save_pic = join('./save_pic')
save_txt = join('./save_txt')

check_dir(save_pic)
check_dir(save_txt)

# 获取图像的路径，以及图像对应框框的标签
imgs_dir = [os.path.join('./background', f) for f in os.listdir('./background') if f.endswith('jpg')]
# labels_dir = [os.path.join('.\\txt', f) for f in os.listdir('txt') if f.endswith('txt')]
labels_dir = hp.replace_labels(imgs_dir)  # 原图上目标对应的标签


# small_imgs_dir = [f.strip() for f in open(join(base_dir, 'small.txt')).readlines()]
small_imgs_dir = [f.strip() for f in open(join('./small.txt')).readlines()]
print(small_imgs_dir)
random.shuffle(small_imgs_dir)


for image_dir, label_dir in zip(imgs_dir, labels_dir):
    print(image_dir, label_dir)
    small_img = []
    for x in range(times):
        if small_imgs_dir == []:
            small_imgs_dir = [f.strip() for f in open(join('./small.txt')).readlines()]
            random.shuffle(small_imgs_dir)
        small_img.append(small_imgs_dir.pop())
    # print("ok")
    aug.copysmallobjects(image_dir, label_dir, save_pic, save_txt, small_img, times, cl_id)
