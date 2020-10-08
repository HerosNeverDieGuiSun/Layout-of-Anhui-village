# _*_coding:utf_8_*_
# 作者  ： zzd
# 创建时间  : 2020/9/24  17:01
# 文件名  ： Main.py
# 内容  :    block处理

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
from numba import jit
import csv



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


# block清洗函数
def block_claen(road_cnts):
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
        while (j < 5):
            if ((a[i][j] == b[j]).all()):
                j = j + 1
            else:
                break;
        if j == 5:
            return i
    return -1


# 向量角度算法
@jit(nopython=True)
def vector_angle(house, road):
    i = 0
    while (i < len(house)):
        house_coordinate = house[i][0]
        count = 0
        for j in range(math.ceil(road.shape[0] / 2)):
            vector1 = house_coordinate - road[j]
            vector1_len = math.sqrt(np.square(vector1[0]).sum())
            flag2 = 0
            k = 0
            while (k < road.shape[0]):
                vector2 = house_coordinate - road[k]
                vector2_len = math.sqrt(np.square(vector2[0]).sum())
                result = (vector1 * vector2).sum() / (vector2_len * vector1_len)
                if (result < -0.98):
                    flag2 = 1
                    break
                k = k + 1
            if (flag2 == 1):
                count = count + 1
        if (count != math.ceil(road.shape[0] / 2)):
            del house[i]
            i = i - 1
        i = i + 1


# 判断轮廓包围函数
def inhere(data):
    # block清洗:剔除过大或者过小的区间
    block_claen(data[10]['cnts'])
    k = 0
    block_all_data = []
    data_copy = copy.deepcopy(data)
    while (k < len(data_copy[10]['cnts'])):
        road = data_copy[10]['cnts'][k]
        block_data = []
        block_house = []
        type = []
        for i in range(10):
            house = data_copy[i]['cnts'].copy()
            vector_angle(house, road)
            block_house.append(house.copy())
            if (len(block_house[i]) != 0):
                block_data = block_data + block_house[i]

        for i in range(10):
            if (len(block_house[i]) != 0):
                for j in range(len(block_house[i])):
                    if(len(block_house[i][j]) != 1):
                        type.append(i)
                    index = find_index(data_copy[4]['cnts'], block_house[i][j])
                    if index != -1:
                        del data_copy[i]['cnts'][index]
        iter = 0
        length = len(block_data)
        while(iter<len(block_data)):
            if(len(block_data[iter]) == 1):
                del block_data[iter]
            else:
                iter = iter + 1

        block_data.insert(0, np.array(type))
        block_data.append(road)
        # 空block清除
        if(len(block_data[0])!=0):
            block_all_data.append(block_data)
        k = k + 1

    return block_all_data


# 提取轮廓函数
def select_range(hsv):
    ball_color = 'yellow'

    color_dist = {
        # 日形
        "purple_dark": {'Lower': np.array([135, 255, 110]), 'Upper': np.array([150, 255, 153])},
        # H形
        "yellow_dark": {'Lower': np.array([25, 245, 70]), 'Upper': np.array([32, 255, 104])},
        # 凹形
        "red_dark": {'Lower': np.array([0, 160, 140]), 'Upper': np.array([2, 173, 155])},
        # 回形
        "blue_dark": {'Lower': np.array([92, 110, 190]), 'Upper': np.array([95, 130, 210])},
        # 无天井有院子
        "green_dark": {'Lower': np.array([64, 110, 190]), 'Upper': np.array([68, 124, 210])},
        # 无天井无院子形
        "black": {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 46])},
        # 书院
        "green": {'Lower': np.array([40, 200, 200]), 'Upper': np.array([45, 255, 255])},
        # 祠堂
        'red': {'Lower': np.array([0, 250, 200]), 'Upper': np.array([2, 255, 220])},
        # 公园
        'blue': {'Lower': np.array([85, 250, 250]), 'Upper': np.array([95, 255, 255])},
        # 商业建筑
        "purple": {'Lower': np.array([145, 127, 245]), 'Upper': np.array([150, 210, 255])},
        # 道路
        "yellow": {'Lower': np.array([27, 250, 250]), 'Upper': np.array([32, 255, 255])},

    }
    data = {}
    i = 0
    for key in color_dist.keys():
        # 选取范围
        inRange_hsv = cv2.inRange(hsv, color_dist[key]['Lower'], color_dist[key]['Upper'])
        # 提取轮廓
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
        # cnts_arr = np.array(cnts)
        temp = {'inRange_hsv': inRange_hsv, 'cnts': cnts}
        data[i] = temp
        i = i + 1

    return data


# csv文件写入函数
def write_csv(block_all_data):
    # 数据结构调整
    for i in range(len(block_all_data)):
        for j in range(len(block_all_data[i])):
            block_all_data[i][j] = block_all_data[i][j].tolist()
    # 文件写入
    with open('5.csv', 'w') as file:
        csvwriter = csv.writer(file, lineterminator='\n')
        csvwriter.writerows(block_all_data)


if __name__ == "__main__":
    # 读取图片
    frame = cv2.imread("5.png")
    # 高斯模糊
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # 转化成HSV图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 提取轮廓
    data = select_range(hsv)
    # 判断建筑是否在block内
    block_all_data = inhere(data)
    # 提取所有有效block数据
    write_csv(block_all_data)

    # 图像展示
    road_data = data[10]
    # showimg(frame, block_all_data[3])
