# _*_coding:utf_8_*_
# 作者  ： zzd
# 创建时间  : 2020/9/24  17:01
# 文件名  ： block_deal.py
# 内容  :    block处理

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
from numba import jit
import csv
import file_process as fp


# block清洗函数
# 将不符合规定的block删除
def block_clean(road_cnts):
    i = 0
    while (i < len(road_cnts)):
        if (road_cnts[i].shape[0] < 10 or road_cnts[i].shape[0] > 2000):
            del road_cnts[i]
            i = i - 1
        i = i + 1


# 匹配函数，返回index
def find_index(a, b):
    for i in range(len(a)):
        j = 0
        while (j < 1):
            if ((a[i][j] == b[j]).all()):
                j = j + 1
            else:
                break;
        if j == 1:
            return i
    return -1


# 向量角度算法
@jit(nopython=True)
def vector_angle(house, road):
    i = 0
    while (i < len(house)):
        # 房屋的单个像素坐标
        house_coordinate = house[i][0]
        # 统计匹配个数
        count = 0
        # 只计算一半的道路数据
        for j in range(math.ceil(road.shape[0] / 2)):
            # 设置第一个向量及其长度
            vector1 = house_coordinate - road[j]
            vector1_len = math.sqrt(np.square(vector1[0]).sum())
            flag2 = 0
            k = 0
            while (k < road.shape[0]):
                # 设置第二个向量及其长度
                vector2 = house_coordinate - road[k]
                vector2_len = math.sqrt(np.square(vector2[0]).sum())
                # 返回cos(theta)
                result = (vector1 * vector2).sum() / (vector2_len * vector1_len)
                if (result < -0.98):
                    flag2 = 1
                    break
                k = k + 1
            # 存在对应向量，统计个数
            if (flag2 == 1):
                count = count + 1
        # 如果个数不匹配，证明house不在block中，删除数据
        if (count != math.ceil(road.shape[0] / 2)):
            del house[i]
            i = i - 1
        i = i + 1


# 判断轮廓包围函数
def inhere(data):
    # block清洗:剔除过大或者过小的区间
    block_clean(data[10]['cnts'])
    # block迭代次数
    k = 0
    # 存储全部block数据
    block_all_data = []
    # 对data数据进行深拷贝
    data_copy = copy.deepcopy(data)
    while (k < len(data_copy[10]['cnts'])):
        # block中道路轮廓数据
        road = data_copy[10]['cnts'][k]
        # 单个block数据
        block_data = []
        # 单个block中房子的数据
        block_house = []
        # 房子类型
        type = []
        for i in range(10):
            # 从data中提取一个房子数据
            house = data_copy[i]['cnts'].copy()
            if (len(house) != 0):
                # 进行向量角度算法，如果该房子不在该block中则删除
                vector_angle(house, road)
                # 将house中剩余部分加入block_house中
                block_house.append(house.copy())
                if (len(block_house[i]) != 0):
                    # 将房子数据存储到block_data中
                    block_data = block_data + block_house[i]

            else:
                # 无house则加入空数据
                block_house.append([])

        for i in range(10):
            if (len(block_house[i]) != 0):
                for j in range(len(block_house[i])):
                    # 确认每一个house的类型
                    type.append(i)
                    # 找到house在原数据中所在的index
                    index = find_index(data_copy[i]['cnts'], block_house[i][j])
                    if index != -1:
                        # 将已经确认在block的屋子从原数据中删除，加快寻找速度
                        del data_copy[i]['cnts'][index]

        # 删除block中单个像素的house
        iter = 0
        while (iter < len(block_data)):
            if (len(block_data[iter]) == 1):
                del block_data[iter]
            else:
                iter = iter + 1
        # 在开头加入类型
        block_data.insert(0, np.array(type))
        # 在结尾加入道路信息
        block_data.append(road)

        # 空block清除
        if (len(block_data[0]) != 0):
            # 将单个block信息添加到block_all_data中
            block_all_data.append(block_data)
        k = k + 1

    return block_all_data


if __name__ == "__main__":
    data = fp.label_read_img('1')
    # 判断建筑是否在block内
    block_all_data = inhere(data)

    # 提取所有有效block数据
    fp.cnts_write_csv(block_all_data, '1')

    # 图像展示
    # road_data = data[10]

