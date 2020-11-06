# -*- coding: utf-8 -*- 
# @Time : 2020/11/6 20:37 
# @Author : zzd 
# @File : data_process.py 
# @desc:


import csv
import json
import numpy as np


# str转array坐标函数
def toarray(str):
    # 转成list
    temp = json.loads(str)
    arr = np.array(temp)
    return arr


def todict(str):
    a = '{"name" : "john", "gender" : "male", "age": 28}'
    b = json.loads(a)
    str = delchar(str)
    temp = json.loads(str)
    print()


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
    print()
