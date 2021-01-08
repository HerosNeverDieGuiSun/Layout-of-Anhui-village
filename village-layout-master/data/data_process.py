import csv
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import cv2


# str转array坐标函数
def toarray(str):
    arr = eval(str)
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
    temp['vercoordinate'] = toarray(temp['vercoordinate'])
    temp['side'] = toarray(temp['side'])
    temp['area'] = float(temp['area'])
    temp['angle'] = int(temp['angle'])
    temp['dist_house'] = toarray(temp['dist_house'])
    temp['dist_road'] = float(temp['dist_road'])
    return temp


# 删除'\'元素
def delchar(str):
    a = str.replace('\'', "\"")
    return a


# 从block_cnts.csv中读取数据
def cnts_read_csv():
    # 设置文件路径
    CSV_FILE_PATH = '../NEW_CSV/block_cnts.csv'
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
def info_read_csv():
    # 设置文件路径
    CSV_FILE_PATH = '../NEW_CSV/block_info.csv'
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


# 从NEW_CSV的village_dis.csv中读取每个block到村中心的距离数据
def vdis_read_csv():
    # 设置文件路径
    CSV_FILE_PATH = '../NEW_CSV/village_dis.csv'
    vdis = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            vdis.append(line)

    for i in range(len(vdis[0])):
        vdis[0][i] = int(vdis[0][i])
    return vdis[0]


# 从cnts_block_categories.csv中读取数据
def cnts_block_categories_read_csv():
    # 设置文件路径
    CSV_FILE_PATH = '../frequency_files/cnts_block_categories.csv'

    # 定义存储数据机构
    data = []

    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            data.append(line)
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = toarray(data[i][j])

    return data


# 从info_block_categories.csv中读取数据
def info_block_categories_read_csv():

    # 设置文件路径
    CSV_FILE_PATH = '../frequency_files/info_block_categories.csv'
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
            if j != 0:
                str1 = delchar(data[i][j])
                data[i][j] = json.loads(str1)
    return data


# 合并各个村子的CSV文件
def village_merge():
    # 创建保存合并后的csv 文件夹
    save_new_csv_dir = '../NEW_CSV'
    if not os.path.exists(save_new_csv_dir):
        os.mkdir(save_new_csv_dir)

    # 定义存储数据机构
    cnts_data = []
    info_data = []
    vdis_data = []

    for i in range(4):
        # 设置文件路径
        CSV_FILE_PATH_cnts = '../CSV/' + str(i + 1) + '_block_cnts.csv'
        CSV_FILE_PATH_info = '../CSV/' + str(i + 1) + '_block_info.csv'
        CSV_FILE_PATH_vds = '../CSV/' + str(i + 1) + '_village_dis.csv'
        # 数据读取
        with open(CSV_FILE_PATH_cnts, 'r') as cnts_f:
            cnts_file = csv.reader(cnts_f)
            for line in cnts_file:
                cnts_data.append(line)
        with open(CSV_FILE_PATH_info, 'r') as info_f:
            info_file = csv.reader(info_f)
            for line in info_file:
                info_data.append(line)
        with open(CSV_FILE_PATH_vds, 'r') as vdis_f:
            vdis_file = csv.reader(vdis_f)
            for line in vdis_file:
                if i == 0:
                    vdis_data.append(line)
                else:
                    for j in range(len(line)):
                        vdis_data[0].append(line[j])

    with open(save_new_csv_dir + '/block_cnts.csv', 'w') as file:
        csv_writer = csv.writer(file, lineterminator='\n')
        csv_writer.writerows(cnts_data)
    with open(save_new_csv_dir + '/block_info.csv', 'w') as file:
        csv_writer = csv.writer(file, lineterminator='\n')
        csv_writer.writerows(info_data)
    with open(save_new_csv_dir + '/village_dis.csv', 'w') as file:
        csv_writer = csv.writer(file, lineterminator='\n')
        csv_writer.writerows(vdis_data)
    # print()  # 测试