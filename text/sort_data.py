# -*- coding: utf-8 -*- 
# @Time : 2020/10/12 19:44 
# @Author : zl、zzd
# @File : sort_data.py
# @desc:  将csv的记录整理成对应的数据格式

from Distances.Kd_tree import KDTree
import csv
import numpy as np
import json
import sys
import cv2
import math
import file_process as fp
import towards


# 寻找最小矩形包围盒函数
def min_all_rect(data):
    # 包围盒中心点坐标
    box_center = [[] for i in range(len(data))]
    # 包围盒四个顶点坐标
    box_vercoordinate = [[] for i in range(len(data))]
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
            box_vercoordinate[i].append(b_int)
            j = j + 1
    return box_center, box_vercoordinate


# 获取矩形包围盒面积函数
def get_area(side):
    area = side[0] * side[1]
    area = float('%0.3f' % area)
    return area


# 寻找block中两建筑间的最近距离
def shortest_house_dist(box_center):
    allhouse2house_min_dist = []
    for i in range(len(box_center)):
        # 存储一个block中建筑个数的序列
        block_housenum = []
        # 一个block中建筑的中心坐标点集合
        house_center = []
        for l in range(len(box_center[i])):
            block_housenum.append(l)
            house_center.append(box_center[i][l].tolist())
        # 一个block中建筑间最近距离的计算
        temp_house_house_min_dist = []

        for s in range(len(house_center)):
            house_house_min_dist = []
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
            min_dist = float('%0.3f' % min_dist)
            # 存储一个block中每个建筑与其相距最近建筑的距离
            house_house_min_dist.append(nd.split[0])
            house_house_min_dist.append(min_dist)
            temp_house_house_min_dist.append(house_house_min_dist)

            house_center.insert(s, m)  # 重新表示为原block中建筑中心坐标集合
            block_housenum.insert(s, n)  # 重新表示为原block中建筑个数的序列

        allhouse2house_min_dist.append(temp_house_house_min_dist)

    return allhouse2house_min_dist


# 寻找block中的建筑与最近路的距离
def shortest_road_dist(box_center,box_vercoordinate, data):
    all_house_road_min_dist = []
    all_house_towards = []
    y = [0, 1, 2]
    a = y[-1]
    # # 一条block数据
    for i in range(len(data)):
        # 存储block路的坐标点
        road_point = data[i][-1].tolist()
        road_point = flat_list(road_point)
        # 记录block路的坐标点个数序列
        road_point_num = []
        for t in range(len(data[i][-1])):
            road_point_num.append(t)

        # # 一条block数据中的一个建筑
        # 一条block中所有建筑的边长中点坐标集合
        house_side_len_center = [[] for m in range(len(box_vercoordinate[i]))]
        # house_road_min_dist = [[] for n in range(len(box_vercoordinate[i]))]
        house_road_min_dist = []
        house_toward = []
        for l in range(len(box_vercoordinate[i])):
            k = 0

            # # 一个建筑的四条边长的中点坐标集合
            for k in range(len((box_vercoordinate[i][l]))):
                if (k == 3):
                    sidelencenter = (box_vercoordinate[i][l][k] + box_vercoordinate[i][l][0]) / 2
                else:
                    sidelencenter = (box_vercoordinate[i][l][k + 1] + box_vercoordinate[i][l][k]) / 2
                house_side_len_center[l].append(sidelencenter.tolist())

            # 建立 KD Tree
            tree = KDTree()
            tree.build_tree(road_point, road_point_num)
            # KD Tree 搜索最近距离
            min_dist = sys.maxsize
            min_nd = []
            for s in range(len(house_side_len_center[l])):
                nd = tree.nearest_neighbour_search(house_side_len_center[l][s])
                temp = distance(nd.split[0], house_side_len_center[l][s])
                if (min_dist > temp):
                    min_dist = temp
                    min_nd = nd.split[0]

            min_dist = math.sqrt(min_dist)
            min_dist = float('%0.3f' % min_dist)
            house_road_min_dist.append(min_dist)
            house_toward.append(get_angle(min_nd,box_center[i][l]))
        all_house_road_min_dist.append(house_road_min_dist)
        all_house_towards.append(house_toward)

    return all_house_road_min_dist,all_house_towards

# 获取朝向
def get_angle(min_nd,box_center):
    x = abs(min_nd[0]-box_center[0])
    y = abs(min_nd[1]-box_center[1])
    z = math.sqrt(x * x + y * y)
    if (min_nd[0] == box_center[0] and min_nd[1] == box_center[1]):
        angle = 0
    else:
        angle = round(math.asin(y / z) / math.pi * 180)

    if (box_center[0] > min_nd[0] and box_center[1] < min_nd[1]):
        angle = 180 - angle
    elif (box_center[0] > min_nd[0] and box_center[1] > min_nd[1]):
        angle = 180 + angle
    elif (box_center[0] < min_nd[0] and box_center[1] > min_nd[1]):
        angle = 360 - angle
    return angle

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


# 获取长宽
def get_side(vercoordinate):
    side = []
    a = math.sqrt(pow(vercoordinate[0][0] - vercoordinate[1][0], 2) + pow(vercoordinate[0][1] - vercoordinate[1][1], 2))
    a = float('%0.3f' % a)
    b = math.sqrt(pow(vercoordinate[0][0] - vercoordinate[3][0], 2) + pow(vercoordinate[0][1] - vercoordinate[3][1], 2))
    b = float('%0.3f' % b)
    if a > b:
        side.append(a)
        side.append(b)
    else:
        side.append(b)
        side.append(a)
    return side


# 得到每一个block到村子中心的距离
def viliage_dis(data, viliage_center):
    temp = []
    for i in range(len(data)):
        temp.append(str(int(distance(cv2.minAreaRect(data[i][-1])[0], viliage_center))))
    return temp


# 数据整理，生成字典
def sort(data, all_house_towards,shd, srd):
    # 获取最小矩形包围盒中心点坐标及四个顶点坐标
    box_center, box_vercoordinate = min_all_rect(data)
    # 初始化单元格信息
    info = []
    for i in range(len(box_center)):
        block = []
        for j in range(len(box_center[i])):
            cell = {}
            cell['label'] = data[i][0][j]
            cell['center'] = box_center[i][j]
            cell['vercoordinate'] = box_vercoordinate[i][j].tolist()
            cell['side'] = get_side(box_vercoordinate[i][j])
            cell['area'] = get_area(cell['side'])
            cell['angle'] = all_house_towards[i][j]
            cell['dist_house'] = shd[i][j]
            cell['dist_road'] = srd[i][j]
            block.append(cell)
        info.append(block)
    return info


if __name__ == "__main__":
    # 导入csv数据信息
    data = fp.cnts_read_csv('1')

    # fp.show_cnts('2',data)

    # cnts = fp.towards_read_img("2")
    viliage_center = fp.get_viliage_center('1')
    vdis = viliage_dis(data, viliage_center)
    vdis = [vdis]
    # pair = towards.calculate_towards_vector(cnts, data)
    # towards.calculate_towards_angle(pair)
    fp.vdis_write_csv(vdis, '1')

    # # 获取最小矩形包围盒中心点坐标及四个顶点坐标

    fp.vdis_write_csv(vdis, '1')
    #
    (box_center, box_vercoordinate) = min_all_rect(data)
    # 根据中心点坐标获取距离最近的房子
    shd = shortest_house_dist(box_center)
    # 根据矩形包围盒四边中点获取距离最近的路
    srd,all_house_towards = shortest_road_dist(box_center,box_vercoordinate, data)
    #
    info = sort(data, all_house_towards, shd, srd)
    fp.info_write_csv(info, '1')
    # fp.show_rect('1', data)

