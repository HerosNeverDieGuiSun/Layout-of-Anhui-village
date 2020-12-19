# -*- coding: utf-8 -*- 
# @Time : 2020/11/6 20:37 
# @Author : zzd 
# @File : bayesian.py 
# @desc:
from pgmpy.models import BayesianModel
from sklearn.cluster import KMeans
import data_process as dp
import numpy as np
import cv2
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.mixture import GaussianMixture
import os
import copy
import math
from pgmpy.sampling import GibbsSampling
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
from intersected import is_intersected


# 整理信息，得到想要的数据
def sort(filename, info, data, vdis):
    # block面积
    road_area = []
    # 每一个block中房子的数量
    house_num = []
    # 房子类型
    house_type = []
    # 添加数据
    for i in range(len(data)):
        road_area.append(cv2.contourArea(data[i][-1]))
        house_num.append(len(data[i][0]))
        house_type.append(data[i][0])
    # 调整road_area的格式
    road_area = np.array([road_area]).reshape(len(road_area), 1)
    vdis = np.array([vdis]).reshape(len(vdis), 1)
    # 为了克服每次聚类结果不一致的问题，决定将其写入文档中
    if (os.path.exists('gaussian/' + filename + '_road_area.txt')):
        road_area = dp.gaussian_read(filename + '_road_area')
        vdis = dp.gaussian_read(filename + '_vdis')
    else:
        # 使用高斯模糊聚类算法，将面积分为10类
        road_model = GaussianMixture(n_components=5)
        road_area = road_model.fit_predict(road_area)
        # 使用高斯模糊聚类算法，将vdis分为5类
        vdis_model = GaussianMixture(n_components=5)
        vdis = vdis_model.fit_predict(vdis)
        # 写入
        dp.gaussian_write(road_area, filename + '_road_area')
        dp.gaussian_write(vdis, filename + '_vdis')
        # kmeans算法
        # km = KMeans(n_clusters=10).fit(road_area)
        # road_area = km.fit_predict(road_area)
    # bn1_info 信息整理
    bn1_info = []
    for i in range(len(data)):
        item = []
        item.append(road_area[i])
        item.append(vdis[i])
        # 添加房屋个数
        item.append(len(data[i][0]))
        bn1_info.append(item)
    # bn2_info信息整理
    bn2_info = []
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            item = []
            item.append(road_area[i])
            item.append(vdis[i])
            # 添加房屋个数
            item.append(len(data[i][0]))
            item.append(data[i][0][j])
            bn2_info.append(item)
    # for i in range(len(data)):
    #     item = []
    #     item.append(road_area[i])
    #     item.append(len(house_type[i]))
    #     for k in range(20):
    #         item.append(int(0))
    #     for j in range(len(data[i][0])):
    #         item[data[i][0][j] + 2] = 1
    #         item[data[i][0][j] + 12] = item[data[i][0][j] + 12] + 1
    #     all.append(item)
    return bn1_info, bn2_info


# 定义BN，进行测试
def initial_BN(bn1_info, bn2_info, data, info):
    model1 = BayesianModel([('block_area', 'house_num'), ('vdis', 'house_num')])
    df1 = pd.DataFrame(bn1_info, columns=['block_area', 'vdis', 'house_num'])
    model1.fit(df1, estimator=BayesianEstimator, prior_type="BDeu")
    model1_infer = VariableElimination(model1)

    model2 = BayesianModel(
        [('block_area', 'house_num'), ('vdis', 'house_num'), ('block_area', 'most_type'), ('vdis', 'most_type'),
         ('house_num', 'most_type')])
    df2 = pd.DataFrame(bn2_info, columns=['block_area', 'vdis', 'house_num', 'most_type'])
    model2.fit(df2, estimator=BayesianEstimator, prior_type="BDeu")
    model2_infer = VariableElimination(model2)

    # for cpd in model2.get_cpds():
    #     print(cpd)
    bn_info = bn2_info

    # for i in range(5):
    #     for j in range(5):
    house_num = model1_infer.map_query(variables=['house_num'], evidence={'block_area': 1, 'vdis': 4})[
        'house_num']
    most_type = model2_infer.map_query(variables=['most_type'],
                                       evidence={'block_area': 1, 'vdis': 4,
                                                 'house_num': house_num})['most_type']
    guess_list = get_guess_list(data, house_num, most_type)

    direction_list = []

    for i in range(len(guess_list)):
        j = i + 1
        while (j < len(guess_list)):
            direction_list.append(two_house_angle(info, data, guess_list[i], guess_list[j]))
            j = j + 1
    print(guess_list)


# 给定两个类型，返回这两个类型最有可能存在的位置关系
def two_house_angle(info, data, type1, type2):
    train_data = []
    for i in range(len(info)):
        # 获取符合要求的节点对
        index = get_pair(data[i][0], type1, type2)

        for j in range(len(index)):
            angle = get_angle(info[i][index[j][0]]['center'], info[i][index[j][1]]['center'])
            dis = two_house_dis(info[i][index[j][0]], info[i][index[j][1]], index)
            direction = angle2direction(angle)
            # train_data.append([type1, type2, direction,dis])
            train_data.append([type1, type2, direction])

    # model = BayesianModel([('type1', 'direction'), ('type2', 'direction'),('type1', 'dis'),('type2', 'dis')])
    model = BayesianModel([('type1', 'direction'), ('type2', 'direction')])
    # df = pd.DataFrame(train_data, columns=['type1', 'type2', 'direction','dis'])
    df = pd.DataFrame(train_data, columns=['type1', 'type2', 'direction'])
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
    return direction_infer[0]


# 获取两个房子的之间的距离
def two_house_dis(type1, type2, index):
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
        i = min(line[0][1],line[1][1])
        while i <= max(line[0][1],line[1][1]):
            point.append([line[0][0],i])
            i = i + 1
    return point


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


# 获取猜测房子的类型
def get_guess_list(data, house_num, most_type):
    guess_list = [most_type]
    # guess_list = [4, 5]

    train_data = []
    p = 0
    while (p < house_num - 1):
        # 针对data中的每一条数据：
        for i in range(len(data)):
            # 取出一个block中所有房子
            house_list = copy.deepcopy(list(data[i][0]))
            # house_list = [4, 5]
            # 判断他们房屋个数是否一致
            if house_num == len(house_list):
                # 判断guess_list是否是house_list的子集
                if set(guess_list).issubset(set(house_list)):
                    # 获取两个list的差集
                    remain_list = get_remain_list(copy.deepcopy(house_list), copy.deepcopy(guess_list))
                    # 如果存在差集
                    if (len(remain_list) != 0):

                        for j in range(len(remain_list)):
                            # 获取对应数据
                            item = []
                            item.append(house_num)
                            for k in range(len(guess_list)):
                                item.append(guess_list[k])
                            item.append(remain_list[j])
                            train_data.append(item)
                # else:
                # print('2')
        if (len(train_data) != 0):
            bn_tuple = [('house_num', 'guess_type')]
            bn_node = ['house_num']
            for i in range(len(guess_list)):
                bn_tuple.append(('type_' + str(i), 'guess_type'))
                bn_node.append('type_' + str(i))
            bn_node.append('guess_type')

            model = BayesianModel(bn_tuple)
            df = pd.DataFrame(train_data, columns=bn_node)
            model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")
            model_infer = VariableElimination(model)

            # for cpd in model.get_cpds():
            #     print(cpd)

            evidence_dict = {'house_num': train_data[0][0]}
            for i in range(len(train_data[0]) - 2):
                evidence_dict['type_' + str(i)] = train_data[0][i + 1]
            guess_type = model_infer.map_query(variables=['guess_type'],
                                               evidence=evidence_dict)['guess_type']
            guess_list.append(int(guess_type))
        p = p + 1
    return guess_list


# 获取剩余的房子
def get_remain_list(house_list, guess_list):
    i = 0
    while (i < len(guess_list)):
        j = 0
        while (j < len(house_list)):
            if (house_list[j] == guess_list[i]):
                del house_list[j]
                del guess_list[i]
                i = i - 1
                break
            j = j + 1
        if len(guess_list) == 0:
            break
        else:
            i = i + 1
    return house_list


if __name__ == "__main__":
    info = dp.info_read_csv('1')
    vdis = dp.vdis_read_csv('1')
    data = dp.cnts_read_csv('1')
    bn1_info, bn2_info = sort('1', info, data, vdis)
    initial_BN(bn1_info, bn2_info, data, info)
    # dp.showBN(model)
    # print(df)
