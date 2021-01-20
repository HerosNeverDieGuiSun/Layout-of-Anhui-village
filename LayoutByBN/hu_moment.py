# -*- coding: utf-8 -*- 
# @Time : 2021/1/19 16:48 
# @Author : zzd 
# @File : hu_moment.py 
# @desc: 用于hu不变矩的相关处理
import cv2
import math
import numpy as np
import data_process as dp

def moment_deal(cnts):
    block_cnts = []
    for i in range(len(cnts)):
        block_cnts.append(cnts[i][-1])
    cnts2img(block_cnts)

    # 测试Hu不变矩的效果
    test(block_cnts)

    print()

def test(block_cnts):
    # dp.show_cnts('1', block_cnts)
    # similar = []
    # k = 11
    # for i in range(len(block_cnts)):
    #     if i != k :
    #         # similar.append(cal_similar2(humoments[i],humoments[5]))
    #         similar.append(cv2.matchShapes(block_cnts[i], block_cnts[k], 1, 0.0))
    # a = similar.index(min(similar))
    #
    # temp = [block_cnts[k],block_cnts[a]]
    # dp.show_cnts('1', temp)
    # # 提取轮廓
    # color_dist = {"black": {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 46])}}



    cnts = []
    for i in range(6):


        fp = './blockimg/'+str(i+1)+'.png'
        frame = cv2.imread(fp)
        # # 高斯模糊
        # gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # # 转化成HSV图像
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # # 选取范围
        # inRange_hsv = cv2.inRange(hsv, color_dist['black']['Lower'], color_dist['black']['Upper'])
        # # 提取轮廓
        # cnts.append( cv2.findContours(inRange_hsv.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2][0])

        # moments = cv2.moments(img_gray)
        # humoments.append(cv2.HuMoments(moments))
    # # 计算相似性
    similar = []
    for i in range(5):
        # similar.append(cal_similar2(humoments[i],humoments[5]))
        similar.append(cv2.matchShapes(cnts[i], cnts[5], 1, 0.0))

    print()

def cal_similar1(data,target):
    dSigmaST = 0
    dSigmaS = 0
    dSigmaT = 0
    for i in range(7):
        temp = math.fabs(data[i] * target[i])
        dSigmaST = dSigmaST + temp
        dSigmaS = dSigmaS + math.pow(data[i],2)
        dSigmaT = dSigmaT + math.pow(target[i],2)
    dbR = dSigmaST / (math.sqrt(dSigmaS) * math.sqrt(dSigmaT))
    return dbR

def cal_similar2(data,target):
    dbR2 = 0
    temp2 = 0
    temp3 = 0
    for i in range(7):
        temp2 = temp2+math.fabs(data[i] - target[i])
        temp3 = temp3+math.fabs(data[i] + target[i])
    dbR2 = 1 - (temp2 * 1.0) / (temp3)
    return dbR2

def cnts2img (block_cnts):



    print()

