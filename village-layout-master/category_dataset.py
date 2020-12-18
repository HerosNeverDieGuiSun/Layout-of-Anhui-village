# -*- coding: utf-8 -*-
# @Time : 2020/12/17 21:49
# @Author : zl
# @File : category_dataset.py
# @desc:

import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data.rendered import RenderedComposite
import numpy as np
import pdb


class CategoryDataset(Dataset):
    def __init__(self, source, block_category):
        if not os.path.exists(source):
            print('数据文件路径未找到')

        self.block_category = block_category  # 当前block类别
        self.data_dir = [source + '/cnts/{}_blocks'.format(self.block_category), source + '/info/{}_blocks'.format(self.block_category)]
        self.cnts_files = os.listdir(self.data_dir[0])  # 获取当前block类别下的cnts的txt文件目录
        self.info_files = os.listdir(self.data_dir[1])  # 获取当前block类别下的info的txt文件目录
        self.data_len = len(self.cnts_files)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # idx的范围是从0到 len(self)的索引
        txt_name = [self.cnts_files[idx], self.info_files[idx]]
        with open(self.data_dir[0] + '/' + txt_name[0], 'r') as cnts_f:
            cnts = eval(cnts_f.read())
        with open(self.data_dir[1] + '/' + txt_name[1], 'r') as info_f:
            info = eval(info_f.read())
        block_tensor = RenderedComposite(cnts, info).get_composite()
        return block_tensor


# 测试tensor保存图片
def save_channel_img(dest_dir, data_loader):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    for i_batch, sample_batched in enumerate(data_loader):
        block_file = dest_dir + '/{}_block'.format(i_batch)
        if not os.path.exists(block_file):
            os.mkdir(block_file)
        block_file1 = block_file + '/block.png'
        block_file2 = block_file + '/houses.png'
        # print(len(sample_batched[0]))  # 14

        # 对于校正通道后的图像，需要利用plt.imsave()保存
        plt.imsave(block_file1, sample_batched[0][0].squeeze(), cmap='gray')  # block
        plt.imsave(block_file2, sample_batched[0][1].squeeze(), cmap='gray')  # house


if __name__ == "__main__":
    data_dir = './txt_data_divide'
    block_category = 3
    dataset = CategoryDataset(data_dir, block_category)

    # # 测试下数据读取
    # c = [data_dir + '/cnts', data_dir + '/info']
    # for index in range(len(os.listdir(c[0]))):
    #     b = [i + '/{}_{}.txt'.format(str(index), i[-4:]) for i in c]
    #     with open(b[0], 'r') as cnts_f:
    #         cnts = eval(cnts_f.read())
    #         print(cnts)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
       )

    # 显示
    # data_iter = iter(data_loader)
    # data_read = next(data_iter)
    # print(data_read[0].shape)
    # image = plt.imshow(data_read[0][1].squeeze(), cmap='gray')  # 绘图函数imshow()
    # plt.show()  # 显示图像

    # 保存
    # dest_dir = '../channel_img'

    dest_dir = './channel_img_divide'
    save_channel_img(dest_dir, data_loader)
