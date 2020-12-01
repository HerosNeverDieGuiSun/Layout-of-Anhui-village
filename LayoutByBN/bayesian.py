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
def initial_BN(bn1_info, bn2_info, data):
    # model = BayesianModel([('block_area', 'house_num'), ('block_area', 'type_0'),
    #                        ('block_area', 'type_1'), ('block_area', 'type_2'), ('block_area', 'type_3'),
    #                        ('block_area', 'type_4'),
    #                        ('block_area', 'type_5'), ('block_area', 'type_6'), ('block_area', 'type_7'),
    #                        ('block_area', 'type_8'), ('block_area', 'type_9'), ('house_num', 'type_0'),
    #                        ('house_num', 'type_1'), ('house_num', 'type_2'), ('house_num', 'type_3'),
    #                        ('house_num', 'type_4'),
    #                        ('house_num', 'type_5'), ('house_num', 'type_6'), ('house_num', 'type_7'),
    #                        ('house_num', 'type_8'), ('house_num', 'type_9'), ('type_0', 'num_0'),
    #                        ('type_1', 'num_1'), ('type_2', 'num_2'), ('type_3', 'num_3'),
    #                        ('type_4', 'num_4'), ('type_5', 'num_5'), ('type_6', 'num_6'),
    #                        ('type_7', 'num_7'), ('type_8', 'num_8'), ('type_9', 'num_9')])
    model1 = BayesianModel([('block_area', 'house_num'), ('vdis', 'house_num')])
    df1 = pd.DataFrame(bn1_info, columns=['block_area', 'vdis', 'house_num'])
    model1.fit(df1, estimator=BayesianEstimator, prior_type="BDeu")

    model1_infer = VariableElimination(model1)
    # for i in range(10):
    #     for j in range(5):
    #         q = model1_infer.map_query(variables=['house_num'], evidence={'block_area': i, 'vdis': j})
    #         print('block_area=' + str(i) + ' and vdis=' + str(j) + ' and house_num = ' + str(q))
    # for factor in q.values():
    #     print(factor)

    model2 = BayesianModel(
        [('block_area', 'house_num'), ('vdis', 'house_num'), ('block_area', 'most_type'), ('vdis', 'most_type'),
         ('house_num', 'most_type')])
    df2 = pd.DataFrame(bn2_info, columns=['block_area', 'vdis', 'house_num', 'most_type'])
    model2.fit(df2, estimator=BayesianEstimator, prior_type="BDeu")
    model2_infer = VariableElimination(model2)

    # for cpd in model2.get_cpds():
    #     print(cpd)
    bn_info = bn2_info

    for i in range(5):
        for j in range(5):
            house_num = model1_infer.map_query(variables=['house_num'], evidence={'block_area': i, 'vdis': j})[
                'house_num']
            most_type = model2_infer.map_query(variables=['most_type'],
                                               evidence={'block_area': i, 'vdis': j,
                                                         'house_num': house_num})['most_type']
            guess_list = get_guess_list(data, house_num, most_type)
            print(guess_list)


# 获取猜测房子的类型
def get_guess_list(data, house_num, most_type):
    guess_list = [most_type]
    # guess_list = [4, 5]

    train_data = []
    p = 0
    while (p < house_num-1):
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
    initial_BN(bn1_info, bn2_info, data)
    # dp.showBN(model)
    # print(df)
