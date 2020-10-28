# -*- coding: utf-8 -*- 
# @Time : 2020/10/26 16:36 
# @Author : zzd 
# @File : file_process.py 
# @desc: 负责文件读取写入、图像展示等工作

import csv
import numpy as np
import json
import cv2


# block轮廓数据 csv文件写入函数
def cnts_write_csv(block_all_data, filename):
    # 数据结构调整
    for i in range(len(block_all_data)):
        for j in range(len(block_all_data[i])):
            block_all_data[i][j] = block_all_data[i][j].tolist()
    # 文件写入
    # with open('../CSV/1_block_cnts.csv', 'w') as file:
    with open('../CSV/' + filename + '_block_cnts.csv', 'w') as file:
        csvwriter = csv.writer(file, lineterminator='\n')
        csvwriter.writerows(block_all_data)


def info_write_csv(info, filename):
    # 数据结构调整
    for i in range(len(info)):
        for j in range(len(info[i])):
            for key in info[i][j].keys():
                info[i][j][key] = str(info[i][j][key])
    # 文件写入
    with open('../CSV/' + filename + '_block_info.csv', 'w') as file:
        csvwriter = csv.writer(file, lineterminator='\n')
        csvwriter.writerows(info)


# str转array坐标函数
def toarray(str):
    # 转成list
    temp = json.loads(str)
    arr = np.array(temp)
    return arr


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


# 读取房屋标记图片
def label_read_img(filename):
    # 读取图片
    frame = cv2.imread('../Label/' + filename + '.png')
    # 高斯模糊
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # 转化成HSV图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 提取轮廓
    data = house_select_range(hsv)
    return data


# 读取朝向图片
def towards_read_img(filename):
    # 读取图片
    frame = cv2.imread("../Towards/" + filename + ".png")
    # 高斯模糊
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # 转化成HSV图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 提取轮廓
    cnts = towards_select_range(hsv)
    # 图形展示
    # showimg(frame,cnts)
    return cnts


# 清除不合适的红点
def towards_clean_cnts(cnts):
    i = 0
    while (i < len(cnts)):
        if (len(cnts[i]) < 7):
            del cnts[i]
            i = i - 1
        i = i + 1

    return cnts


# 提取朝向轮廓
def towards_select_range(hsv):
    color_dist = {
        # 提取红色块
        "red": {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
    }
    # 选取范围
    inRange_hsv = cv2.inRange(hsv, color_dist["red"]['Lower'], color_dist["red"]['Upper'])
    # 提取轮廓
    cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    cnts = towards_clean_cnts(cnts)
    return cnts


# 提取房屋和道路轮廓函数
def house_select_range(hsv):
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


# 初始化画布
def init_canvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas


# 轮廓图像展示函数
def show_cnts(filename, cnts):
    frame = cv2.imread('../Label/' + filename + '.png')
    x = frame.shape[1]
    y = frame.shape[0]
    # 生成指定大小的画布
    canvas = init_canvas(x, y, color=(255, 255, 255))
    cv2.polylines(canvas, cnts, 1, 0)
    cv2.imshow("frame", canvas)
    cv2.imwrite("canny.jpg", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')


# 展示房屋朝向
def show_pair(filename, pair):
    # 读取图片
    frame = cv2.imread("../Towards/" + filename + ".png")
    x = frame.shape[1]
    y = frame.shape[0]
    # 生成指定大小的画布
    canvas = init_canvas(x, y, color=(255, 255, 255))

    for i in range(len(pair)):
        cv2.line(canvas, (pair[i][0][0], pair[i][0][1]), (pair[i][1][0], pair[i][1][1]), (0, 0, 255), 1)

    cv2.imshow("frame", canvas)
    cv2.imwrite("canny.jpg", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')


# 展示红点和识别不出的房子
def show_error_house(filename, cnts, house):
    # 读取图片
    frame = cv2.imread("../Towards/" + filename + ".png")
    x = frame.shape[1]
    y = frame.shape[0]
    # 生成指定大小的画布
    canvas = init_canvas(x, y, color=(255, 255, 255))
    cv2.polylines(canvas, cnts, 1, 0)
    cv2.polylines(canvas, house, 1, 0)
    cv2.imshow("frame", canvas)
    cv2.imwrite("canny.jpg", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')


# 展示矩形包围盒
def show_rect(filename, data):
    # 读取图片
    frame = cv2.imread("../Towards/" + filename + ".png")
    # 生成指定大小的画布
    x = frame.shape[1]
    y = frame.shape[0]
    canvas = init_canvas(x, y, color=(255, 255, 255))

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


# 寻找最小矩形函数
def min_rect(house):
    # 找到最小矩形，返回中心坐标，长宽，旋转角度
    rect = cv2.minAreaRect(house)
    # 计算矩形四个顶点坐标
    box = cv2.boxPoints(rect)
    # 转化成int
    box = np.int0(box)
    return box
