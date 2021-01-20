# -*- coding: utf-8 -*- 
# @Time : 2020/11/6 20:37 
# @Author : zzd 
# @File : data_process.py 
# @desc:


import csv
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import cv2
from pgmpy.readwrite import BIFReader, BIFWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# str转array坐标函数
def toarray(str):
    # 转成list
    temp = json.loads(str)
    arr = np.array(temp)
    return arr

# str转list
def str2list(str):
    l = []
    if (len(str) == 9):
        a = int(str[1:4])
        b = int(str[5:8])
        l.append(a)
        l.append(b)
    else:
        print('转成list失败了')
    return l

# str转字典
def todict(str):
    str = delchar(str)
    temp = json.loads(str)
    temp['label'] = int(temp['label'])
    temp['center'] = str2list(temp['center'])
    temp['vercoordinate'] = toarray(temp['vercoordinate']).tolist()
    temp['side'] = toarray(temp['side']).tolist()
    temp['area'] = float(temp['area'])
    temp['angle'] = int(temp['angle'])
    temp['dist_house'] = toarray(temp['dist_house']).tolist()
    temp['dist_road'] = float(temp['dist_road'])
    return temp


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


def road_read_csv(filename):
    # 设置文件路径
    CSV_FILE_PATH = '../CSV/' + filename + '_block_cnts.csv'
    # 定义存储数据机构
    data = []
    road_area = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            data.append(line)
    for i in range(len(data)):
        road_area.append(cv2.contourArea(toarray(data[i][-1])))
    return road_area


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
    return data


# vdis的csv文件读取
def vdis_read_csv(filename):
    # 设置文件路径
    CSV_FILE_PATH = '../CSV/' + filename + '_village_dis.csv'
    vdit = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            vdit.append(line)

    for i in range(len(vdit[0])):
        vdit[0][i] = int(vdit[0][i])
    return vdit[0]


# 高斯文件写入
def gaussian_write(data, filename):
    string = ''
    for i in range(len(data)):
        string = string + str(data[i])
    with open('gaussian/' + filename + '.txt', 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(string)


def gaussian_read(filename):
    with open('gaussian/' + filename + '.txt', 'r') as f:
        data = np.array(list(f.read()))
    return data


# 贝叶斯网络图像展示
def showBN(model):
    edges = model.edges()
    G = nx.MultiDiGraph()
    for a, b in edges:
        G.add_edge(a, b)
    nx.draw(G, with_labels=True, edge_color='gray', node_color='skyblue', node_size=100, width=3)
    plt.show()
    print()


# 写入概率图形文件
def write_bif(model, filename):
    if (os.path.exists('./bif/' + filename + '.bif')):
        print("文件已经存在啦")
    else:
        writer = BIFWriter(model)
        writer.write_bif(filename='./bif/' + filename + '.bif')


# 读取概率模型文件
def read_bif(filename):
    if (os.path.exists('./bif/' + filename + '.bif')):
        model = BIFReader('./bif/' + filename + '.bif').get_model()
        return model
    else:
        print("读取bif文件失败")
        return 0

# 字符转int
def str2int(input_list):
    for i in range(len(input_list)):
        input_list[i] = int(input_list[i])
    return input_list


# 初始化画布
def init_canvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas

def save_block(filename,cnts):
    # 生成指定大小的画布
    canvas = init_canvas(1500, 1000, color=(255, 255, 255))


# 轮廓图像展示函数,使用的data数据
def show_cnts(filename, data):
    # frame = cv2.imread('../TestImage/' + filename + '.png')
    # x = frame.shape[1]
    # y = frame.shape[0]
    # 生成指定大小的画布
    canvas = init_canvas(1500, 1200, color=(255, 255, 255))
    cv2.polylines(canvas, data, 1, 0)
    cv2.imshow("frame", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')


# 展示布局
def show_layout(cnts, corner):
    filename = '1'
    frame = cv2.imread('../TestImage/' + filename + '.png')
    x = frame.shape[1]
    y = frame.shape[0]
    # 生成指定大小的画布
    canvas = init_canvas(x, y, color=(255, 255, 255))
    cv2.polylines(canvas, cnts, 1, 0)
    # for i in range(len(corner)):
    #     cv2.rectangle(canvas, corner[i][0], corner[i][2],(0,0,0))
    cv2.imshow("frame", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')

# 展示布局
def show_line(cnts,input_cnts):
    filename = '1'
    frame = cv2.imread('../TestImage/' + filename + '.png')
    x = frame.shape[1]
    y = frame.shape[0]
    # 生成指定大小的画布
    canvas = init_canvas(x, y, color=(255, 255, 255))
    for i in range(len(cnts)):
        if i != len(cnts)-1 :
            cv2.line(canvas, (cnts[i][0][0],cnts[i][0][1]), (cnts[i+1][0][0],cnts[i+1][0][1]), color=(0, 0, 0),thickness=1)
        else:
            cv2.line(canvas, (cnts[i][0][0],cnts[i][0][1]), (cnts[0][0][0],cnts[0][0][1]), color=(0, 0, 0), thickness=1)
    cv2.polylines(canvas, input_cnts, 1, 0)
    cv2.imshow("frame", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('frame')

# 读取测试文件
def read_text_image(filename):
    # 读取图片
    frame = cv2.imread("../TestImage/" + filename + ".png")
    # 高斯模糊
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # 转化成HSV图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_dist = {
        # 提取道路
        "black": {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 46])},
    }
    # 选取范围
    inRange_hsv = cv2.inRange(hsv, color_dist["black"]['Lower'], color_dist["black"]['Upper'])
    # 提取轮廓
    cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    center = cv2.minAreaRect(cnts[0])[0]
    return cnts[0]
