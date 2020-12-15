# -*- coding: utf-8 -*-
# @Time : 2020/12/13 16:10
# @Author : zl
# @File : dataset.py
# @desc:

import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data.rendered import RenderedComposite
import numpy as np


class VillageDataset(Dataset):
    def __init__(self, source):
        if not os.path.exists(source):
            print('数据文件路径未找到')
        self.data_dir = [source + '/cnts', source + '/info']
        self.data_len = len(os.listdir(self.data_dir[0]))

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # idx的范围是从0到 len(self)的索引
        txt_name = [i + '/{}_{}.txt'.format(str(idx), i[-4:]) for i in self.data_dir]
        with open(txt_name[0], 'r') as cnts_f:
            cnts = eval(cnts_f.read())
        with open(txt_name[1], 'r') as info_f:
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
    data_dir = '../txt_data'
    dataset = VillageDataset(data_dir)

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

    # # 显示
    # data_iter = iter(data_loader)
    # data_read = next(data_iter)
    # print(data_read[0].shape)
    # image = plt.imshow(data_read[0][0].squeeze(), cmap='gray')  # 绘图函数imshow()
    # plt.show()  # 显示图像

    # 保存
    dest_dir = '../channel_img'
    save_channel_img(dest_dir, data_loader)



