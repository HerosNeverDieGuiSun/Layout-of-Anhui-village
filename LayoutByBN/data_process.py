# -*- coding: utf-8 -*- 
# @Time : 2020/11/6 20:37 
# @Author : zzd 
# @File : data_process.py 
# @desc:


import csv
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# str转array坐标函数
def toarray(str):
    # 转成list
    temp = json.loads(str)
    arr = np.array(temp)
    return arr


def str2list(str):
    l = []
    if (len(str) == 9):
        a = int(str[1:4])
        b = int(str[5:8])
        l.append(a)
        l.append(b)
    else:
        print('转成list失败了')
    return l


def todict(str):
    str = delchar(str)
    temp = json.loads(str)
    temp['label'] = int(temp['label'])
    temp['center'] = str2list(temp['center'])
    temp['vercoordinate'] = toarray(temp['vercoordinate']).tolist()
    temp['side'] = toarray(temp['side']).tolist()
    temp['area'] = float(temp['area'])
    temp['angle'] = int(temp['angle'])
    temp['dist_house'] = toarray(temp['dist_house']).tolist()
    temp['dist_road'] = float(temp['dist_road'])
    return temp


# 删除'\'元素
def delchar(str):
    a = str.replace('\'', "\"")
    return a


# 从block_cnts.csv中读取数据
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

    # 将所有数据从str 转成array
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = toarray(data[i][j])

    return data


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
        road_area.append(cv2.contourArea(toarray(data[i][-1])))
    return road_area


# 从block_info.csv中读取数据
def info_read_csv(filename):
    # 设置文件路径
    CSV_FILE_PATH = '../CSV/' + filename + '_block_info.csv'
    # 定义存储数据机构
    data = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            data.append(line)
        # 将所有数据从str 转成array
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = todict(data[i][j])
    return data


# vdis的csv文件读取
def vdis_read_csv(filename):
    # 设置文件路径
    CSV_FILE_PATH = '../CSV/' + filename + '_viliage_dis.csv'
    vdit = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            vdit.append(line)

    for i in range(len(vdit[0])):
        vdit[0][i] = int(vdit[0][i])
    return vdit[0]


def gaussian_write(data, filename):
    string = ''
    for i in range(len(data)):
        string = string + str(data[i])
    with open('gaussian/' + filename + '.txt', 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(string)


def gaussian_read(filename):
    with open('gaussian/' + filename + '.txt', 'r') as f:
        data = np.array(list(f.read()))
    return data

# 贝叶斯网络图像展示
def showBN(model):
    edges = model.edges()
    G = nx.MultiDiGraph()
    for a, b in edges:
        G.add_edge(a, b)
    nx.draw(G, with_labels=True, edge_color='gray', node_color='skyblue', node_size=100, width=3)
    plt.show()
    print()
