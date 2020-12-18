# -*- coding: utf-8 -*- 
# @Time : 2020/10/21 16:50 
# @Author : zzd 
# @File : towards.py
# @desc: 计算图形朝向信息

import cv2
import numpy as np
import csv
import json
import sys
import file_process as fp
import math


# 计算叉乘
def get_cross(pt1, pt2, pt):
    return (pt2[0] - pt1[0]) * (pt[1] - pt1[1]) - (pt[0] - pt1[0]) * (pt2[1] - pt1[1])


# 判断是否在矩形内
def in_matrix(square, pt):
    if ((get_cross(square[0], square[1], pt) * get_cross(square[2], square[3], pt)) >= 0 and
            (get_cross(square[1], square[2], pt) * get_cross(square[3], square[0], pt)) >= 0):
        return True
    else:
        return False


# 寻找最小矩形函数
def min_rect(house):
    # 找到最小矩形，返回中心坐标，长宽，旋转角度
    rect = cv2.minAreaRect(house)
    # 计算矩形四个顶点坐标
    box = cv2.boxPoints(rect)
    # 转化成int
    box = np.int0(box)
    return box, np.int0(rect[0])


# 检查函数
def examination(pair):
    for i in range(len(pair)):
        j = i + 1
        while (j < len(pair)):
            if (pair[i][1][0] == pair[j][1][0] and pair[i][1][1] == pair[j][1][1]):
                print("出事了" + "i=" + str(i) + "and j = " + str(j))
            j = j + 1


# 计算房屋中心和朝向中心的距离
def calculate_dist(house_center, center):
    return pow(house_center[0] - center[0], 2) + pow(house_center[1] - center[1], 2)


# 寻找与house最近的朝向坐标点
def min_dist(house_center, towards_center):
    # 设置最大值
    temp = sys.maxsize
    label = 0
    for p in range(len(towards_center)):
        dist = calculate_dist(house_center, towards_center[p])
        if (dist < temp):
            temp = dist
            label = p
    return label


# 计算data中屋子的数量
def house_num(data, cnts):
    count = 0
    for i in range(len(data)):
        count = count + len(data[i]) - 2
    print("房屋的数量为:" + str(count) + "  朝向点的数量为：" + str(len(cnts)))


# 计算朝向
def calculate_towards_vector(cnts, data):
    towards_center = []
    for i in range(len(cnts)):
        # 找到最小矩形，返回中心坐标，长宽，旋转角度
        rect = cv2.minAreaRect(cnts[i])
        # 转成int
        rect_int = np.int0(rect[0])
        # 加入list
        towards_center.append(rect_int)

    pair = []
    for i in range(len(data)):
        j = 1
        while (j < len(data[i]) - 1):
            # 提取house数据
            house = data[i][j]
            # 返回房屋四角坐标和中心点坐标
            box, house_center = min_rect(house)
            j = j + 1
            k = 0
            while (k < len(towards_center)):
                # 判断朝向中心点是否在矩形中
                if (in_matrix(box, towards_center[k]) == True):
                    # 加入节点对中
                    pair.append([house_center, towards_center[k]])
                    break
                else:
                    k = k + 1

            # 如果朝向点没有在包围盒中，则寻找其最近朝向点
            if (k == len(towards_center)):
                label = min_dist(house_center, towards_center)
                pair.append([house_center, towards_center[label]])

    examination(pair)
    return pair


def calc_angle(x1, y1, x2, y2):
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    z = math.sqrt(x * x + y * y)
    if (x1 == x2 and y1 == y2):
        angle = 0
    else:
        angle = round(math.asin(y / z) / math.pi * 180)

    if (x1 > x2 and y1 < y2):
        angle = 180 - angle
    elif (x1 > x2 and y1 > y2):
        angle = 180 + angle
    elif (x1 < x2 and y1 > y2):
        angle = 360 - angle

    return angle


def calculate_towards_angle(pair):
    for i in range(len(pair)):
        angle = calc_angle(pair[i][0][0], pair[i][0][1], pair[i][1][0], pair[i][1][1])
        pair[i].append(angle)
    return pair


def get_angle(center, pair):
    for i in range(len(pair)):

        if(center[0] == pair[i][0][0] and center[1] == pair[i][0][1]):
            return pair[i][2]
    print('节点对找不到内容呀，出错了')

if __name__ == "__main__":
    cnts = fp.towards_read_img("2")
    data = fp.cnts_read_csv("2")
    house_num(data, cnts)
    pair = calculate_towards_vector(cnts, data)
    calculate_towards_angle(pair)
    t = get_angle([555, 734], pair)
    print()
