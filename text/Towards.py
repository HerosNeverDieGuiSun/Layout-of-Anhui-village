# -*- coding: utf-8 -*- 
# @Time : 2020/10/21 16:50 
# @Author : zzd 
# @File : Towards.py 
# @desc: 计算图形朝向信息

import cv2
import numpy as np
import csv
import json
import sys


# 初始化画布
def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas


# 图像展示函数
def showimg(frame, cnts):
    x = frame.shape[1]
    y = frame.shape[0]
    # 生成指定大小的画布
    canvas = InitCanvas(x, y, color=(255, 255, 255))
    cv2.polylines(canvas, cnts, 1, 0)
    cv2.imshow("frame", canvas)
    cv2.imwrite("canny.jpg", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')


# 清除不合适的红点
def clean_cnts(cnts):
    i = 0
    while (i < len(cnts)):
        if (len(cnts[i]) < 7):
            del cnts[i]
            i = i - 1
        i = i + 1

    return cnts


# str转array坐标函数
def toarray(str):
    temp = json.loads(str)
    arr = np.array(temp)
    return arr


# 提取轮廓
def select_range(hsv):
    color_dist = {
        # 提取红色块
        "red": {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
    }
    # 选取范围
    inRange_hsv = cv2.inRange(hsv, color_dist["red"]['Lower'], color_dist["red"]['Upper'])
    # 提取轮廓
    cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    cnts = clean_cnts(cnts)
    return cnts


# 读取朝向文件
def read_towards_img(filename):
    # 读取图片
    frame = cv2.imread("../Towards/" + filename + ".png")
    # 高斯模糊
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # 转化成HSV图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 提取轮廓
    cnts = select_range(hsv)
    # 图形展示
    # showimg(frame,cnts)
    return cnts


# 读取block的轮廓数据
def read_block_csv(filename):
    CSV_FILE_PATH = "../CSV/" + filename + '_block_cnts.csv'
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


# 展示红点和识别不出的房子
def show(cnts, house):
    # 读取图片
    frame = cv2.imread("../Towards/" + "1" + ".png")
    x = frame.shape[1]
    y = frame.shape[0]
    # 生成指定大小的画布
    canvas = InitCanvas(x, y, color=(255, 255, 255))
    cv2.polylines(canvas, cnts, 1, 0)
    cv2.polylines(canvas, house, 1, 0)
    cv2.imshow("frame", canvas)
    cv2.imwrite("canny.jpg", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')


def examination(pair):
    for i in range(len(pair)):
        j = i + 1
        while (j < len(pair)):
            if (pair[i][1][0] == pair[j][1][0] and pair[i][1][1] == pair[j][1][1]):
                print("出事了" + "i=" + str(i) + "and j = " + str(j))
            j = j + 1


# 展示红点和识别不出的房子
def show(pair):
    # 读取图片
    frame = cv2.imread("../Towards/" + "1" + ".png")
    x = frame.shape[1]
    y = frame.shape[0]
    # 生成指定大小的画布
    canvas = InitCanvas(x, y, color=(255, 255, 255))

    for i in range(len(pair)):
        cv2.line(canvas, (pair[i][0][0], pair[i][0][1]), (pair[i][1][0], pair[i][1][1]), (0, 0, 255), 1)

    cv2.imshow("frame", canvas)
    cv2.imwrite("canny.jpg", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')


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
def calculate_towards(cnts, data):
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
    show(pair)


if __name__ == "__main__":
    cnts = read_towards_img("1")
    data = read_block_csv("1")
    house_num(data, cnts)
    calculate_towards(cnts, data)
