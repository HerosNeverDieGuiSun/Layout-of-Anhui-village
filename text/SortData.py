# -*- coding: utf-8 -*- 
# @Time : 2020/10/12 19:44 
# @Author : zzd、zl
# @File : SortData.py 
# @desc:  将csv的记录整理成对应的数据格式

from Distances.Kd_tree import KDTree
import csv
import numpy as np
import json
import sys
import cv2
import math


# 初始化画布
def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas


# str转array坐标函数
def toarray(str):
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


# 寻找最小矩形包围盒函数
def min_all_rect(data):
    # 包围盒中心点坐标
    box_center = [[] for i in range(len(data))]
    # 包围盒四个顶点坐标
    box_Vercoordinate = [[] for i in range(len(data))]
    # 计算包围盒中心点坐标、四顶点坐标、(宽,高)，旋转角
    for i in range(len(data)):
        j = 1
        while j != len(data[i]) - 1:
            house = data[i][j]
            # 找到最小矩形包围盒，返回（中心(x,y), (宽,高), 旋转角度）
            box = cv2.minAreaRect(house)
            # 提取包围盒中心点坐标(x,y)
            box_center[i].append(np.int0((box[0])))
            # 获取包围盒四个顶点坐标
            b = cv2.boxPoints(box)
            # 转化成int
            b_int = np.int0(b)
            box_Vercoordinate[i].append(b_int)
            j = j + 1
    return box_center, box_Vercoordinate


    # 读取图片
    # frame = cv2.imread("../Lable/1.png")
    # showimg(frame, house, box)


# 寻找最小矩形函数
def min_rect(house):
    # 找到最小矩形，返回中心坐标，长宽，旋转角度
    rect = cv2.minAreaRect(house)
    # 计算矩形四个顶点坐标
    box = cv2.boxPoints(rect)
    # 转化成int
    box = np.int0(box)
    return box



def showing_all(frame, data):
    # 生成指定大小的画布
    x = frame.shape[1]
    y = frame.shape[0]
    canvas = InitCanvas(x, y, color=(255, 255, 255))

    for i in range(len(data)):
        j = 1
        while (j < len(data[i])):
            if (j < len(data[i]) - 1):
                box = min_rect(data[i][j])
                # 绘制矩形
                cv2.line(canvas, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 0, 255), 1)
                cv2.line(canvas, (box[0][0], box[0][1]), (box[3][0], box[3][1]), (0, 0, 255), 1)
                cv2.line(canvas, (box[1][0], box[1][1]), (box[2][0], box[2][1]), (0, 0, 255), 1)
                cv2.line(canvas, (box[2][0], box[2][1]), (box[3][0], box[3][1]), (0, 0, 255), 1)

            # 绘制房屋
            cv2.polylines(canvas, data[i][j], 1, 0)
            j = j + 1

    cv2.imshow("frame", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')


# 获取矩形包围盒面积函数
def get_area(cnt):
    area = cv2.contourArea(cnt)
    print(area)


# 寻找block中两建筑间的最近距离
def shortestHouse_dist(box_center):
    allhouseHouse_min_dist = []
    for i in range(len(box_center)):
        # 存储一个block中建筑个数的序列
        block_housenum = []
        # 一个block中建筑的中心坐标点集合
        house_center = []
        for l in range(len(box_center[i])):
            block_housenum.append(l)
            house_center.append(box_center[i][l].tolist())
        # 一个block中建筑间最近距离的计算
        houseHouse_min_dist = []
        for s in range(len(house_center)):
            m = house_center[s]  # 保存将要移除的建筑中心坐标
            n = block_housenum[s]  # 保存将要移除的建筑序列序号
            house_center.remove(house_center[s])  # 移除当前建筑中心坐标
            block_housenum.remove(block_housenum[s])  # 此时个数序列总值减一

            # 建立 KD Tree
            tree = KDTree()
            tree.build_tree(house_center, block_housenum)
            # KD Tree 搜索最近距离
            nd = tree.nearest_neighbour_search(m)
            min_dist = math.sqrt(distance(nd.split[0], m))
            # 存储一个block中每个建筑与其相距最近建筑的距离
            houseHouse_min_dist.append(min_dist)

            house_center.insert(s, m)  # 重新表示为原block中建筑中心坐标集合
            block_housenum.insert(s, n)  # 重新表示为原block中建筑个数的序列

        allhouseHouse_min_dist.append(houseHouse_min_dist)

    return allhouseHouse_min_dist


# 寻找block中的建筑与最近路的距离
def shortestRoad_dist(box_Vercoordinate, data):
    allHouseRoad_min_dist = []
    y = [0,1,2]
    a = y[-1]
    # # 一条block数据
    for i in range(len(data)):
        # 存储block路的坐标点
        road_point = data[i][-1].tolist()
        road_point = flat_list(road_point)
        # 记录block路的坐标点个数序列
        road_pointNum = []
        for t in range(len(data[i][-1])):
            road_pointNum.append(t)

        # # 一条block数据中的一个建筑
        # 一条block中所有建筑的边长中点坐标集合
        house_sideLenCenter = [[] for m in range(len(box_Vercoordinate[i]))]
        # houseRoad_min_dist = [[] for n in range(len(box_Vercoordinate[i]))]
        houseRoad_min_dist = []
        for l in range(len(box_Vercoordinate[i])):
            k = 0

            # # 一个建筑的四条边长的中点坐标集合
            for k in range(len((box_Vercoordinate[i][l]))):
                if (k == len((box_Vercoordinate[i][l])) - 1):
                    sidelencenter = (box_Vercoordinate[i][l][k] + box_Vercoordinate[i][l][0]) / 2
                else:
                    sidelencenter = (box_Vercoordinate[i][l][k+1] + box_Vercoordinate[i][l][k])/2
                house_sideLenCenter[l].append(sidelencenter.tolist())

            # 建立 KD Tree
            tree = KDTree()
            tree.build_tree(road_point, road_pointNum)
            # KD Tree 搜索最近距离
            min_dist = sys.maxsize
            for s in range(len(house_sideLenCenter[l])):
                nd = tree.nearest_neighbour_search(house_sideLenCenter[l][s])
                temp = distance(nd.split[0], house_sideLenCenter[l][s])
                if (min_dist > temp):
                    min_dist = temp
            min_dist = math.sqrt(min_dist)
            houseRoad_min_dist.append(min_dist)
        allHouseRoad_min_dist.append(houseRoad_min_dist)

    return allHouseRoad_min_dist


# 计算两点间的距离
def distance(point1, point2):
    return pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2)

# 压平list
def flat_list(road_point):
    for i in range(len(road_point)):
        temp = road_point[0][0]
        road_point.extend([temp])
        road_point.remove(road_point[0])
    return road_point


if __name__ == "__main__":
    # a()
    # 导入csv数据信息
    data = read_csv('../CSV/1_block_cnts')
    # 获取最小矩形包围盒中心点坐标及四个顶点坐标
    (box_center, box_Vercoordinate) = min_all_rect(data)
    # 根据中心点坐标获取距离最近的房子
    shd = shortestHouse_dist(box_center)
    # 根据矩形包围盒四边中点获取距离最近的路
    srd = shortestRoad_dist(box_Vercoordinate, data)

    get_area(data[28][7])
    frame = cv2.imread("../Lable/1.png")

    showing_all(frame, data)
    print()


