# -*- coding: utf-8 -*- 
# @Time : 2020/10/12 19:44 
# @Author : zzd 
# @File : SortData.py 
# @desc:  将csv的记录整理成对应的数据格式

from Distances.Kd_tree import KDTree
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
    cv2.line(canvas, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 255, 0), 1)
    cv2.line(canvas, (box[0][0], box[0][1]), (box[3][0], box[3][1]), (0, 255, 0), 1)
    cv2.line(canvas, (box[1][0], box[1][1]), (box[2][0], box[2][1]), (0, 255, 0), 1)
    cv2.line(canvas, (box[2][0], box[2][1]), (box[3][0], box[3][1]), (0, 255, 0), 1)
    # 绘制房屋
    cv2.polylines(canvas, house, 1, 0)
    cv2.imshow("frame", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')



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
    return box_center

    # 读取图片
    # frame = cv2.imread("5.png")
    # showimg(frame, house, box)

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
                cv2.line(canvas, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 255, 0), 1)
                cv2.line(canvas, (box[0][0], box[0][1]), (box[3][0], box[3][1]), (0, 255, 0), 1)
                cv2.line(canvas, (box[1][0], box[1][1]), (box[2][0], box[2][1]), (0, 255, 0), 1)
                cv2.line(canvas, (box[2][0], box[2][1]), (box[3][0], box[3][1]), (0, 255, 0), 1)

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



# 获取面积函数
def get_area(cnt):
    area = cv2.contourArea(cnt)
    print(area)


# 寻找最短距离函数
def Shortest_dist(box_center,data):
    for i in range(len(data)):
        # 存储每个block中建筑个数序列
        block_housenum = []
        # block中建筑的中心坐标点集合
        house_center = []
        for l in range(len(box_center[i])):
            block_housenum.append(l)
            house_center.append(box_center[i][l].tolist())
        # 建立 KD Tree
        tree = KDTree()
        tree.build_tree(house_center, block_housenum)
        j = 0
        while j < len(box_center[i]):
            # block中每个建筑的中心坐标点
            each_house_center = box_center[i][j].tolist()
            # KD Tree 搜索
            nd = tree.nearest_neighbour_search(each_house_center)
            j = j + 1


if __name__ == "__main__":

    # 导入csv数据信息
    data = read_csv('./5')
    # 获取最小矩形包围盒中心点坐标
    box_center = min_all_rect(data)
    # 根据中心点坐标获取距离最近的房子
    Shortest_dist(box_center, data)

    get_area(data[28][7])
    frame = cv2.imread("5.png")
    showing_all(frame, data)
    print()


