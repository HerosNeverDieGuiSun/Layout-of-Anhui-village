# -*- coding: utf-8 -*-
# @Time : 2020/12/15 20:00
# @Author : zl
# @File : csv_transform.py
# @desc:


from data import data_process as dp
import os
from PIL import Image, ImageDraw
import numpy as np

# save_txt_dir = '../txt_data'

save_txt_dir = '../txt_data_divide'
num_block_categories = 5  # 已知划分的block类别数

# def create_txt_data(cnts, info, vdis, start_num, des_dir):
#     """
#     将block的数据依次存到一个个txt中
#     Parameters
#     ----------
#     cnts：cnts数据
#     info：info数据
#     vdis：vdis数据
#     start_num：block的txt文件的id从0开始
#     des_dir：目标文件夹
#     """
#
#     if not os.path.exists(des_dir):
#         os.mkdir(des_dir)
#     cnts_dir = des_dir + '/cnts'
#     info_dir = des_dir + '/info'
#     vdis_dir = des_dir + '/vdis'
#     if not os.path.exists(cnts_dir):
#         os.mkdir(cnts_dir)
#     if not os.path.exists(info_dir):
#         os.mkdir(info_dir)
#     if not os.path.exists(vdis_dir):
#         os.mkdir(vdis_dir)
#
#     for i in range(len(info)):
#         temp_cnts = cnts[i]
#         temp_info = info[i]
#         temp_vdis = vdis[i]
#
#         with open(cnts_dir + '/' + str(start_num) + '_cnts.txt', "w") as cnts_f:
#             cnts_f.write(str(temp_cnts))
#         with open(info_dir + '/' + str(start_num) + '_info.txt', "w") as info_f:
#             info_f.write(str(temp_info))
#         with open(vdis_dir + '/' + str(start_num) + '_vdis.txt', "w") as vdis_f:
#             vdis_f.write(str(temp_vdis))
#
#         start_num += 1


def create_txt_data_divide(cnts_block_cate, info_block_cate, vdis, start_num, des_dir):
    """
    将block的数据依次存到一个个txt中
    Parameters
    ----------
    cnts：cnts数据
    info：info数据
    vdis：vdis数据
    start_num：block的txt文件的id从0开始
    des_dir：目标文件夹
    """

    if not os.path.exists(des_dir):
        os.mkdir(des_dir)
    cnts_dir = des_dir + '/cnts'
    info_dir = des_dir + '/info'
    vdis_dir = des_dir + '/vdis'
    if not os.path.exists(cnts_dir):
        os.mkdir(cnts_dir)
    if not os.path.exists(info_dir):
        os.mkdir(info_dir)
    if not os.path.exists(vdis_dir):
        os.mkdir(vdis_dir)

    for i in range(num_block_categories):
        cnts_blocks_dir = des_dir + '/cnts' + '/' + str(i) + '_blocks'
        info_blocks_dir = des_dir + '/info' + '/' + str(i) + '_blocks'
        if not os.path.exists(cnts_blocks_dir):
            os.mkdir(cnts_blocks_dir)
        if not os.path.exists(info_blocks_dir):
            os.mkdir(info_blocks_dir)

        for j in range(len(cnts_block_cate)):
            if cnts_block_cate[j][0] == i:
                temp_cnts = cnts_block_cate[j][1:]  # cnts中不写入第一列的block标识符
                temp_info = info_block_cate[j][1:]  # info中不写入第一列的block标识符
                temp_vdis = vdis[j]

                with open(cnts_blocks_dir + '/' + str(start_num) + '_cnts_' + str(i) + '.txt', "w") as cnts_b_f:
                    cnts_b_f.write(str(temp_cnts))
                with open(info_blocks_dir + '/' + str(start_num) + '_info_' + str(i) + '.txt', "w") as info_b_f:
                    info_b_f.write(str(temp_info))
                with open(vdis_dir + '/' + str(start_num) + '_vdis.txt', "w") as vdis_f:
                    vdis_f.write(str(temp_vdis))

                start_num += 1

if __name__ == "__main__":
    # # 读取原始的csv数据
    # cnts = dp.cnts_read_csv('1')
    # info = dp.info_read_csv('1')
    # vdis = dp.vdis_read_csv('1')
    # # 获取每个block的txt
    # create_txt_data(cnts, info, vdis, 0, save_txt_dir)

    # # 读取划分block类别后的csv数据：
    # 数据来自于frequency_files里面的1_cnts_block_categories.csv和1_info_block_categories.csv
    cnts_block_cate = dp.cnts_block_categories_read_csv('1')
    info_block_cate = dp.info_block_categories_read_csv('1')
    vdis = dp.vdis_read_csv('1')
    # 获取划分后的每个block类别的txt
    create_txt_data_divide(cnts_block_cate, info_block_cate, vdis, 0, save_txt_dir)
