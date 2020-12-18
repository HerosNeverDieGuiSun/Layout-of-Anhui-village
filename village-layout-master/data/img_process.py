# -*- coding: utf-8 -*-
# @Time : 2020/12/08 22:00
# @Author : zl
# @File : img_process.py
# @desc:

from data import data_process as dp
import numpy as np
import cv2
import os

# 初始化画布
def init_canvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas


# 轮廓图像展示函数
def show_block(filename, cnts_data, label):
    dest_dir = f"../Img_block"
    # 如果没有block_img文件夹则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    frame = cv2.imread('../LABEL/' + filename + '.png')
    x = frame.shape[1]
    y = frame.shape[0]
    # 生成指定原始图片大小的画布
    canvas = init_canvas(x, y, color=(255, 255, 255))

    # 获取并绘制house包围盒轮廓
    j = 1
    while j < len(cnts_data):
        if j < len(cnts_data) - 1:
            box = min_rect(cnts_data[j])
            # 绘制矩形
            cv2.line(canvas, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 0, 255), 1)
            cv2.line(canvas, (box[0][0], box[0][1]), (box[3][0], box[3][1]), (0, 0, 255), 1)
            cv2.line(canvas, (box[1][0], box[1][1]), (box[2][0], box[2][1]), (0, 0, 255), 1)
            cv2.line(canvas, (box[2][0], box[2][1]), (box[3][0], box[3][1]), (0, 0, 255), 1)
        j = j + 1

    # 绘制当前的block轮廓
    cv2.polylines(canvas, cnts_data[-1], 1, 0)

    # cv2.imshow("frame", canvas)
    cv2.imwrite(dest_dir + '/canny' + str(label) + '.jpg', canvas)
    # cv2.waitKey(0)
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


if __name__ == '__main__':
    # 其他文件已改动，该文件代码停止使用
    a = dp.cnts_read_csv('1')
    for i in range(len(a)):
        show_block('1', a[i], i)
