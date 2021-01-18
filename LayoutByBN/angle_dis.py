# -*- coding: utf-8 -*- 
# @Time : 2020/12/24 21:11 
# @Author : zzd 
# @File : angle_dis.py 
# @desc:  获取两个房子之间的角度和距离关系

# 给定两个类型，返回这两个类型最有可能存在的位置关系
from pgmpy.models import BayesianModel
from sklearn.mixture import GaussianMixture
import numpy as np
import math
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
import pandas as pd
from intersected import is_intersected


def two_house_angle_dis(info, data, type1, type2):
    train_data = []
    for i in range(len(info)):
        # 获取符合要求的节点对
        index = get_pair(data[i][0], type1, type2)

        for j in range(len(index)):
            angle = get_angle(info[i][index[j][0]]['center'], info[i][index[j][1]]['center'])
            dis = two_house_dis(info[i][index[j][0]], info[i][index[j][1]])
            direction = angle2direction(angle)
            train_data.append([type1, type2, direction, dis])
            # train_data.append([type1, type2, direction])
    train_data, dis_mean = dis2gaussian(train_data)

    model = BayesianModel([('type1', 'direction'), ('type2', 'direction'), ('type1', 'dis'), ('type2', 'dis')])
    # model = BayesianModel([('type1', 'direction'), ('type2', 'direction')])
    df = pd.DataFrame(train_data, columns=['type1', 'type2', 'direction', 'dis'])
    # df = pd.DataFrame(train_data, columns=['type1', 'type2', 'direction'])
    model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")
    model_infer = VariableElimination(model)
    # direction = model_infer.map_query(variables=['direction'], evidence={'type1': type1, 'type2': type2})[
    #     'direction']
    inference = BayesianModelSampling(model)
    evidence = [State(var='type1', state=type1), State(var='type2', state=type2)]
    direction_infer = inference.rejection_sample(evidence=evidence, size=1, return_type='dataframe')

    type1_cols = direction_infer['type1']
    direction_infer = direction_infer.drop('type1', axis=1)
    direction_infer.insert(0, 'type1', type1_cols)

    direction_infer = direction_infer.values.tolist()

    direction_infer[0][2] = dis_mean[direction_infer[0][2]][0]

    return direction_infer[0]


def save_decimal(input):
    for i in input:
        i[0] = round(i[0], 2)
    return input


# 距离聚类函数
def dis2gaussian(train_data):
    dis_gaussian = GaussianMixture(n_components=5, random_state=0)
    dis_temp = []
    for i in range(len(train_data)):
        dis_temp.append(train_data[i][3])

    dis_temp = np.array([dis_temp]).reshape(len(dis_temp), 1)
    dis_temp = dis_gaussian.fit_predict(dis_temp)
    for i in range(len(train_data)):
        train_data[i][3] = dis_temp[i]
    dis_mean = save_decimal(dis_gaussian.means_)
    return train_data, dis_mean


# 获取两个房子的之间的距离
def two_house_dis(type1, type2):
    # # 分别计算两矩形中心点在X轴和Y轴的距离
    # dx = abs(type1['center'][0] - type2['center'][0])
    # dy = abs(type1['center'][1] - type2['center'][1])

    # a = is_intersected([1,1],[2,2],[1,-1],[-1,1])
    line1 = search_edge(type1, type2, type1)
    line2 = search_edge(type1, type2, type2)
    A1, B1, C1 = generate_equation(line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    A2, B2, C2 = generate_equation(line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    point_list1 = generate_point(A1, B1, C1, line1)
    point_list2 = generate_point(A2, B2, C2, line2)
    min_dis = get_distance(point_list1[0], point_list2[0])
    for i in range(len(point_list1)):
        for j in range(len(point_list2)):
            dis = get_distance(point_list1[i], point_list2[j])
            if dis < min_dis:
                min_dis = dis

    return min_dis


# 获取角度
def get_angle(min_nd, box_center):
    x = abs(min_nd[0] - box_center[0])
    y = abs(min_nd[1] - box_center[1])
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


# 获取两个点之间的距离
def get_distance(point1, point2):
    x = abs(point1[0] - point2[0])
    y = abs(point1[1] - point2[1])
    dis = round(math.sqrt(x * x + y * y), 2)
    return dis


def generate_point(A, B, C, line):
    point = []
    if B != 0:
        i = min(line[0][0], line[1][0])
        while i <= max(line[0][0], line[1][0]):
            y = int((-C - A * i) / B)
            point.append([i, y])
            i = i + 1
    else:
        i = min(line[0][1], line[1][1])
        while i <= max(line[0][1], line[1][1]):
            point.append([line[0][0], i])
            i = i + 1
    return point


# 生成线性函数
def generate_equation(first_x, first_y, second_x, second_y):
    # 一般式 Ax+By+C=0
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    return A, B, C


# 寻找最近的一条边
def search_edge(type1, type2, replace):
    for i in range(4):
        if i < 3:
            flag = is_intersected(type1['center'], type2['center'], replace['vercoordinate'][i],
                                  replace['vercoordinate'][i + 1])
        else:
            flag = is_intersected(type1['center'], type2['center'], replace['vercoordinate'][i],
                                  replace['vercoordinate'][0])
        if flag != 0:
            return flag


# 给定一组标签，返回符合要求的标签对
def get_pair(label, type1, type2):
    label_list = []
    pair = []
    index_temp = []
    index = []
    for i in range(len(label)):
        j = i + 1
        while (j < len(label)):
            temp = []
            label_list.append([label[i], label[j]])
            index_temp.append([i, j])
            j = j + 1
    for i in range(len(label_list)):
        if type1 in label_list[i]:
            if type2 in label_list[i]:
                pair.append(label_list[i])
                index.append(index_temp[i])
    return index


# 由坐标转换成朝向
def angle2direction(angle):
    if angle >= 45 and angle < 135:
        return 'N'
    elif angle >= 135 and angle < 225:
        return 'W'
    elif angle >= 225 and angle < 315:
        return 'S'
    else:
        return 'E'
