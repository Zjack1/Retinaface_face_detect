import torch
from itertools import product as product
import numpy as np
from math import ceil
from data import cfg_mnet, cfg_re50
# f=[20,20]
# a = (1, 2, 3)
# b = ('A', 'B', 'C')
# c = product(range(f[0]), range(f[1]))
# for elem in c:
#     print(elem)
#
#

class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']   # 先验框基础边长，有三层特征层，所以有三种基础边长[[16, 32], [32, 64], [64, 128]]
        self.steps = cfg['steps'] # 三个特征层相对于原图resize的倍数 [8, 16, 32]
        self.clip = cfg['clip']  # 是否在先验框输出之后，clip（修剪）到0-1之间
        self.image_size = image_size
        # feature_maps : [[20, 20], [10, 10], [5, 5]]
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):  # k=0,f=[20,20]、k=1, f=[10,10]、k=2, f=[5,5]
            min_sizes = self.min_sizes[k]  # k = 0,[16, 32],k=1,[32,64],k=2,[64,128]
            for i, j in product(range(f[0]), range(f[1])): # i = [0~19] , j = [0~19]
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]  # 归一化的宽高
                    s_ky = min_size / self.image_size[0]
                    # 特征图的size是除以了step,这里求dense_cx又将他乘回来，x是特征图上的每个点，乘step后，
                    # 就回到了原图(640,640)上的坐标，再将该值除以原图的尺寸，也就是在原图上的位置比例，必定是小于1的数。
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]  # 中心宽高形式的box

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cfg = cfg_mnet
# priorbox = PriorBox(cfg, image_size=(160, 160))  # 实例化一个先验框对象
# priors = priorbox.forward()  # 对象里面的方法（获取先验框坐标以及数量信息）
# priors = priors.to(device)
# prior_data = priors.data