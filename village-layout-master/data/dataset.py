# -*- coding: utf-8 -*-
# @Time : 2020/11/30 20:40
# @Author : zl
# @File : dataset.py
# @desc:

import csv
import json
import numpy as np
from tensorflow.python.ops.clustering_ops import KMeans
from sklearn.cluster import KMeans
import os
import cv2


# str转array坐标函数
def to_array(str):
    # 转成list
    temp = json.loads(str)
    arr = np.array(temp)
    return arr


# 从CSV的_block_cnts.csv中读取所有block数据
def cnts_read_csv(filename):
    # 设置文件路径
    CSV_FILE_PATH = '../CSV/' + filename + '_block_cnts.csv'
    # 定义存储数据机构
    data = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            data.append(line)
    # 将所有数据从str转成array
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = to_array(data[i][j])
    return data


# 读取_block_cnts.csv中的所有的road数据
def road_read_csv(filename):
    # 设置文件路径
    CSV_FILE_PATH = '../CSV/' + filename + '_block_cnts.csv'
    # 定义存储数据机构
    data = []
    road_area = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            data.append(line)
    for i in range(len(data)):
        road_area.append(cv2.contourArea(to_array(data[i][-1])))
    return road_area


# 根据block面积大小将其划分为5个类别
def road_area_divide(data, road_area):
    num_block_categories = 5
    dest_dir = f"../block_categories"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # 使用KMeans聚类：按照block面积来划分类别
    real_road_area = np.array([road_area]).reshape(len(road_area), 1)
    km = KMeans(n_clusters=num_block_categories).fit(real_road_area)  # 分为5个类别
    label_road_area = (km.fit_predict(real_road_area)).tolist()
    # 将每个类别的block数据分别存入对应类别csv文件
    for i in range(num_block_categories):
        block_categories = []
        for j in range(len(label_road_area)):
            if i == label_road_area[j]:
                temp = data[j]
                block_categories.append(temp)
                with open(dest_dir + '/block_categories_' + str(i) + '.csv', 'w') as file:
                    csv_writer = csv.writer(file, lineterminator='\n')
                    csv_writer.writerows(block_categories)

def get_house_categories_frequency(block_categories_label):
    # 设置文件路径
    CSV_FILE_PATH = '../block_categories/'+'/block_categories_' + block_categories_label + '.csv'




if __name__ == '__main__':
    # 测试
    a = cnts_read_csv('1')
    b = road_read_csv('1')
    road_area_divide(a, b)

