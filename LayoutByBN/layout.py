# -*- coding: utf-8 -*- 
# @Time : 2021/1/7 20:19 
# @Author : zzd 
# @File : layout.py 
# @desc: 布局

import data_process as dp
import cv2
import numpy as np
import random
import sys
import math
from relation_graph import Relation_node
import queue
import copy


def initialize_block(road_model, vdis_model):
    cnts = dp.read_text_image('1')
    # dp.show_cnts('1',cnts[0])
    block_area = [cv2.contourArea(cnts)]
    block_area = np.array([block_area]).reshape(len(block_area), 1)
    block_area = road_model.predict(block_area)
    return cnts, block_area[0], 3


def guess_layout(input_cnts, input_area):
    direction_list, guess_list, side_guess = get_fake_data()

    flag = True
    while (flag):
        temp = 0
        point = get_point(input_cnts)
        nodelist = initialization_relationship(direction_list, guess_list, side_guess)
        update_coordinate(point, nodelist)
        corner = []
        for i in range(len(nodelist)):
            corner.append(get_corner(nodelist[i].pos, side_guess[i]))
        for i in range(len(corner)):
            for j in range(4):
                temp = temp + int(cv2.pointPolygonTest(input_cnts, corner[i][j], False))
        if temp == len(nodelist) * 4:
            flag = False
        else:
            # dp.show_layout(input_cnts, corner)
            for i in range(1,3):
                angle = 90 * i
                nodelist2, corner2 = rotate_coordinate(math.radians(angle), point, copy.deepcopy(nodelist),
                                                   copy.deepcopy(corner))
                temp2 = 0
                # dp.show_layout(input_cnts, corner2)
                for i in range(len(corner2)):
                    for j in range(4):
                        temp2 = temp2 + int(cv2.pointPolygonTest(input_cnts, corner2[i][j], False))
                if temp2 == len(nodelist) * 4:
                    corner = corner2
                    nodelist = nodelist2
                    flag = False
                    break
    dp.show_layout(input_cnts, corner)
    return corner

# 优化布局，未做完
def optimization_layout(corner,input_cnts):
    a = cv2.approxPolyDP(input_cnts,5,True)
    dp.show_line(a,input_cnts)
    print()

# 旋转坐标
def rotate_coordinate(angle, point, nodelist, corner):
    # 旋转nodelist的坐标
    for i in range(len(nodelist)):
        sRotatex, sRotatey = srotate(angle, nodelist[i].pos[0], nodelist[i].pos[1], point[0], point[1])
        nodelist[i].pos = [sRotatex, sRotatey]
    # 旋转corner的坐标
    for i in range(len(corner)):
        for j in range(4):
            sRotatex, sRotatey = srotate(angle, corner[i][j][0], corner[i][j][1], point[0], point[1])
            corner[i][j] = (sRotatex, sRotatey)
    return nodelist, corner


# 绕pointx,pointy顺时针旋转
def srotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
    sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
    return int(sRotatex), int(sRotatey)


# 更新坐标
def update_coordinate(point, nodelist):
    for i in range(len(nodelist)):
        nodelist[i].pos[0] = nodelist[i].pos[0] + point[0]
        nodelist[i].pos[1] = nodelist[i].pos[1] + point[1]


# 初始化它们之间的关系
def initialization_relationship(direction_list, guess_list, side_guess):
    nodelist = []
    for i in range(len(guess_list)):
        node = Relation_node(guess_list[i], i, 0)
        nodelist.append(node)
    nodelist[0].change_selected(1)
    nodelist[0].change_pos([0, 0])
    q = queue.Queue()
    q.put(nodelist[0])
    while not q.empty():
        node = q.get()
        index = node.guess_index
        start, end = get_s_e(len(guess_list), index)
        i = start
        after_index = index + 1
        while (i <= end):
            if (node.add_toward(nodelist[after_index], direction_list[i][3])):
                set_pos(node, nodelist[after_index], side_guess[node.guess_index], side_guess[after_index],
                        direction_list[i][3])
                nodelist[after_index].change_selected(1)
                q.put(nodelist[after_index])
                after_index = after_index + 1
            i = i + 1
    return nodelist


# 更新node结点位置
def set_pos(node1, node2, node1_side, node2_side, toward):
    if toward == 'N':
        pos = [node1.pos[0], node1.pos[1] + math.ceil((node1_side[3] / 2 + node2_side[3] / 2))]
        node2.change_pos(pos)
    elif toward == 'S':
        pos = [node1.pos[0], node1.pos[1] - math.ceil((node1_side[3] / 2) + (node2_side[3] / 2))]
        node2.change_pos(pos)
    elif toward == 'E':
        pos = [node1.pos[0] + math.ceil((node1_side[2] / 2) + (node2_side[2] / 2)), node1.pos[1]]
        node2.change_pos(pos)
    elif toward == 'W':
        pos = [node1.pos[0] - math.ceil((node1_side[2] / 2) + (node2_side[2] / 2)), node1.pos[1]]
        node2.change_pos(pos)


# 获取开始位置和结束位置
def get_s_e(length, index):
    if index == 0:
        start = 0
        end = length - 2
    else:
        start = 0
        end = 0
        i = 1
        while (i <= index):
            start = start + length - i
            i = i + 1
        start = start
        j = 1
        while (j <= index + 1):
            end = end + length - j
            j = j + 1
        end = end - 1
    return start, end


# 假数据
def get_fake_data():
    direction_list = [[1, 9, 46.06, 'S'], [1, 9, 95.78, 'E'], [1, 1, 9.32, 'N'], [1, 5, 48.38, 'E'], [9, 9, 5.16, 'N'],
                      [9, 1, 82.13, 'N'], [9, 5, 4.91, 'W'], [9, 1, 46.06, 'N'], [9, 5, 4.91, 'W'], [1, 5, 48.38, 'E']]
    guess_list = [1, 9, 9, 1, 5]
    side_guess = [[5, 1, 8.53, 5.38], [5, 9, 12.64, 10.53], [5, 9, 25.96, 10.53], [5, 1, 8.53, 5.38],
                  [5, 5, 8.53, 5.38]]
    return direction_list, guess_list, side_guess


# 获取四周顶点
def get_corner(point, side):
    corner1 = (int(point[0] - side[2] / 2), int(point[1] + side[3] / 2))
    corner2 = (int(point[0] + side[2] / 2), int(point[1] + side[3] / 2))
    corner3 = (int(point[0] + side[2] / 2), int(point[1] - side[3] / 2))
    corner4 = (int(point[0] - side[2] / 2), int(point[1] - side[3] / 2))
    corner = [corner1, corner2, corner3, corner4]
    return corner


# 获取猜测的点
def get_point(input_cnts):
    flag = True
    x_max, x_min, y_max, y_min = get_max_min(input_cnts)
    x = 0
    y = 0
    while (flag):
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        if (cv2.pointPolygonTest(input_cnts, (x, y), False) == 1):
            flag = False
    return (x, y)


# 获取最大值和最小值
def get_max_min(input_cnts):
    x_max = 0
    x_min = sys.maxsize
    y_max = 0
    y_min = sys.maxsize
    for item in input_cnts:
        temp_x = item[0][0]
        temp_y = item[0][1]
        if x_max < temp_x:
            x_max = temp_x
        if x_min > temp_x:
            x_min = temp_x
        if y_max < temp_y:
            y_max = temp_y
        if y_min > temp_y:
            y_min = temp_y

    return x_max, x_min, y_max, y_min


if __name__ == "__main__":
    info = dp.info_read_csv('1')
    # initialize_block()
