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
from angle_dis import two_house_angle_dis
from house_size import get_house_size
from house_size import train_house_size_model
import layout as lay
from hu_moment import moment_deal

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
    # if (os.path.exists('gaussian/' + filename + '_road_area.txt')):
    #     road_area = dp.gaussian_read(filename + '_road_area')
    #     vdis = dp.gaussian_read(filename + '_vdis')
    # else:
    # 使用高斯模糊聚类算法，将面积分为5类
    road_model = GaussianMixture(n_components=5, random_state=0)
    road_area = road_model.fit_predict(road_area)
    # 使用高斯模糊聚类算法，将vdis分为5类
    vdis_model = GaussianMixture(n_components=5, random_state=0)
    vdis = vdis_model.fit_predict(vdis)
    # # 写入
    # dp.gaussian_write(road_area, filename + '_road_area')
    # dp.gaussian_write(vdis, filename + '_vdis')
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
    return bn1_info, bn2_info, road_model, vdis_model


# 定义BN，进行测试
def initial_BN(bn1_info, bn2_info, data, info):
    model1 = BayesianModel([('block_area', 'house_num'), ('vdis', 'house_num')])
    df1 = pd.DataFrame(bn1_info, columns=['block_area', 'vdis', 'house_num'])
    model1.fit(df1, estimator=BayesianEstimator, prior_type="BDeu")
    dp.write_bif(model1, 'house_num')

    model2 = BayesianModel(
        [('block_area', 'house_num'), ('vdis', 'house_num'), ('block_area', 'most_type'), ('vdis', 'most_type'),
         ('house_num', 'most_type')])
    df2 = pd.DataFrame(bn2_info, columns=['block_area', 'vdis', 'house_num', 'most_type'])
    model2.fit(df2, estimator=BayesianEstimator, prior_type="BDeu")
    dp.write_bif(model2, 'most_type')


def guess(info, input_area, input_vdis):
    model1 = dp.read_bif('house_num')
    model2 = dp.read_bif('most_type')
    model1_infer = VariableElimination(model1)
    model2_infer = VariableElimination(model2)
    length_mean, width_mean = train_house_size_model(info)

    # 采样部分
    # 模型1：
    evidence1 = [State(var='block_area', state=str(input_area)), State(var='vdis', state=str(input_vdis))]
    data1_infer = model_sample(model1, evidence1)
    house_num = data1_infer['house_num']
    evidence2 = [State(var='block_area', state=str(input_area)), State(var='vdis', state=str(input_vdis)),
                 State(var='house_num', state=house_num)]
    data2_infer = model_sample(model2, evidence2)
    most_type = data2_infer['most_type']
    guess_list = get_guess_list(data, int(house_num), int(most_type))
    direction_list = []
    for i in range(len(guess_list)):
        j = i + 1
        while (j < len(guess_list)):
            direction_list.append(two_house_angle_dis(info, data, guess_list[i], guess_list[j]))
            j = j + 1
    print()
    # 模型2：
    # 固定输出部分
    # house_num = model1_infer.map_query(variables=['house_num'], evidence={'block_area': 1, 'vdis': 4})[
    #     'house_num']
    # most_type = model2_infer.map_query(variables=['most_type'],
    #                                    evidence={'block_area': 1, 'vdis': 4,
    #                                              'house_num': house_num})['most_type']
    # guess_list = get_guess_list(data, house_num, most_type)
    #
    # direction_list = []
    #
    # for i in range(len(guess_list)):
    #     j = i + 1
    #     while (j < len(guess_list)):
    #         direction_list.append(two_house_angle_dis(info, data, guess_list[i], guess_list[j]))
    #         j = j + 1

    side_guess = get_house_size(guess_list, length_mean, width_mean)

    return guess_list, side_guess, direction_list


def model_sample(model, evidence):
    inference = BayesianModelSampling(model)
    data_infer = inference.rejection_sample(evidence=evidence, size=1, return_type='dataframe')
    return data_infer.iloc[0]


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
            # model_infer = VariableElimination(model)

            # for cpd in model.get_cpds():
            #     print(cpd)
            evidence = [State(var='house_num', state=train_data[0][0])]
            # evidence_dict = {'house_num': train_data[0][0]}
            for i in range(len(train_data[0]) - 2):
                evidence.append(State(var='type_' + str(i), state=train_data[0][i + 1]))
                # evidence_dict['type_' + str(i)] = train_data[0][i + 1]
            data_infer = model_sample(model, evidence)
            guess_type = data_infer['guess_type']
            # guess_type = model_infer.map_query(variables=['guess_type'],
            #                                    evidence=evidence_dict)['guess_type']
            guess_list.append(int(guess_type))
        else:
            print()
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

    moment_deal(data)

    bn1_info, bn2_info, road_model, vdis_model = sort('1', info, data, vdis)
    input_cnts, input_area, input_vdis = lay.initialize_block(road_model, vdis_model)
    # initial_BN(bn1_info, bn2_info, data, info)
    # guess_list, side_guess, direction_list = guess(info, input_area, input_vdis)
    # lay.guess_layout(input_cnts, input_area, guess_list, side_guess,direction_list)
    corner = lay.guess_layout(input_cnts, input_area)
    lay.optimization_layout(corner,input_cnts)
    # dp.showBN(model)

    # print(df)
