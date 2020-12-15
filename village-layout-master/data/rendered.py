# -*- coding: utf-8 -*-
# @Time : 2020/12/13 14:32
# @Author : zl
# @File : rendered.py
# @desc:

import torch
from PIL import Image, ImageDraw
import numpy as np
import os
from torchvision import transforms
import pdb

import matplotlib.pyplot as plt


class RenderedComposite:
    def __init__(self, cnts, info, size=256, categories_num=10):
        self.cnts = cnts
        self.info = info
        self.size = size  # 暂定尺寸为256×256
        self.categories_num = categories_num  # 房子类别数为10

        self.block_mask = torch.zeros((self.size, self.size))  # block mask
        self.house_mask = torch.zeros((self.size, self.size))  # house mask
        self.categories_map = torch.zeros((self.categories_num, self.size, self.size))  # categories map

        # 方向(待添加)
        self.sin_map = torch.zeros((self.size, self.size))  # sinθ张量
        self.cos_map = torch.zeros((self.size, self.size))  # cosθ张量

        # 用于后面的坐标转换处理
        self.xmax, self.xmin, self.ymax, self.ymin = self.analyze_size()
        self.xtimes, self.ytimes = self.size/(self.xmax-self.xmin), self.size/(self.ymax-self.ymin)

        # 获取 mask
        self.add_block_mask()
        self.add_categories_and_house_map()

    def analyze_size(self):
        x_pos = []
        y_pos = []
        for i in self.cnts[-1]:
            for pos in i:
                x_pos.append(pos[0])
                y_pos.append(pos[1])
        return max(x_pos), min(x_pos), max(y_pos), min(y_pos)

    def add_block_mask(self):
        block_shape = []
        for i in self.cnts[-1]:
            for pos in i:
                trans_x, trans_y = int((pos[0] - self.xmin) * self.xtimes), int((pos[1] - self.ymin) * self.ytimes)
                block_shape.append((trans_x, trans_y))
        img = Image.new('L', (self.size, self.size))    # 创建给定size的灰度图像
        img_block = ImageDraw.Draw(img)                 # 创建一个可以在给定img上绘图的对象
        img_block.polygon(block_shape, fill='white')    # 绘制一个多边形
        img = np.asarray(img)                           # 将img转换成array
        block_img_tensor = torch.tensor(img)            # 将img数据转换为tensor
        self.block_mask = block_img_tensor

    def add_categories_map(self, index, vcd):
        house_shape = [((i[0] - self.xmin) * self.xtimes, (i[1] - self.ymin) * self.ytimes) for i in vcd]
        img = Image.new('L', (self.size, self.size))
        img_house = ImageDraw.Draw(img)
        img_house.polygon(house_shape, fill='white')
        img = np.asarray(img)
        img_tensor = torch.tensor(img) / 255            # 暂时将有房屋设为1
        self.categories_map[index] += img_tensor

    def add_categories_and_house_map(self):

        for i in self.info:
            self.add_categories_map(i['label'], i['vercoordinate'])
        self.house_mask = self.categories_map.sum(axis=0)

    def get_composite(self):
        """
        创建多通道视图表示,用作网络模型输入
        当前通道顺序：
            -0: block_mask
            -1: house_mask
            -2,3: sin and cos of the angle of rotation
            -4~13: single category channel for 10 house categories
        Parameters
        ----------

        """

        composite = torch.zeros((self.categories_num+4, self.size, self.size))
        composite[0] = self.block_mask
        composite[1] = self.house_mask
        composite[2] = self.sin_map
        composite[3] = self.cos_map
        for i in range(self.categories_num):
            composite[i+4] = self.categories_map[i]

        return composite


if __name__ == "__main__":

    # 测试rendered代码
    data_dir = '../txt_data'
    if not os.path.exists(data_dir):
        print('数据文件路径未找到')
    with open(data_dir + '/cnts' + '/0_cnts.txt') as cnts_f:
        cnts = eval(cnts_f.read())
    with open(data_dir + '/info' + '/0_info.txt') as info_f:
        info = eval(info_f.read())
    r = RenderedComposite(cnts, info).get_composite()
    # print(r)
    # print(r.shape)

    # 测试：显示tensor的图像
    print(r[0])
    for i in range(r.size(0)):
        plt.imshow(r[i], cmap='gray')
        plt.show()
