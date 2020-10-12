# -*- coding: utf-8 -*- 
# @Time : 2020/10/12 19:44 
# @Author : zzd 
# @File : SortData.py 
# @desc:  将csv的记录整理成对应的数据格式

import pandas as pd
import csv
import numpy as np
import json
import cv2


# 初始化画布
def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas


# str转array坐标函数
def toarray(str):
    # # 获取数据个数
    # num = str.count(']], [[',0, len(str))
    # num = num +1

    # 转成list
    temp = json.loads(str)
    arr = np.array(temp)
    return arr


# 从csv中读取数据
def read_csv(filename):
    # 设置文件路径
    CSV_FILE_PATH = filename + '.csv'
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


def showimg(frame, house, box):
    # 生成指定大小的画布
    x = frame.shape[1]
    y = frame.shape[0]
    canvas = InitCanvas(x, y, color=(255, 255, 255))
    # 绘制矩形
    cv2.line(canvas, (254, 602), (227, 594), (0, 255, 0), 1)
    cv2.line(canvas, (254, 602), (271, 546), (0, 255, 0), 1)
    cv2.line(canvas, (227, 594), (244, 538), (0, 255, 0), 1)
    cv2.line(canvas, (244, 538), (271, 546), (0, 255, 0), 1)
    # 绘制房屋
    cv2.polylines(canvas, house, 1, 0)
    cv2.imshow("frame", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')


# 寻找最小矩形函数
def min_rect(house):
    # 找到最小矩形，返回中心坐标，长宽，旋转角度
    rect = cv2.minAreaRect(house)
    # 计算矩形四个顶点坐标
    box = cv2.boxPoints(rect)
    # 转化成int
    box = np.int0(box)
    # 读取图片
    frame = cv2.imread("5.png")
    showimg(frame, house, box)


if __name__ == "__main__":
    data = read_csv('5')
