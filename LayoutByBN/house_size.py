# -*- coding: utf-8 -*- 
# @Time : 2020/12/24 21:18 
# @Author : zzd 
# @File : house_size.py 
# @desc: 估算房屋的尺寸

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
import data_process as dp


def train_house_size_model(info):
    train_data = get_train_data(info)
    model = BayesianModel([('house_num', 'length'), ('house_num', 'width'), ('type', 'length'), ('type', 'width')])
    df = pd.DataFrame(train_data, columns=['house_num', 'type', 'length', 'width'])
    model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")
    model_infer = VariableElimination(model)
    dp.write_bif(model, 'house_size')


def get_house_size(info,guess_list):
    model2 = dp.read_bif('house_size')
    # model2.name = ''
    # model2.graph.clear()
    # train_data = get_train_data(info)
    # model = BayesianModel([('house_num', 'length'), ('house_num', 'width'), ('type', 'length'), ('type', 'width')])
    # df = pd.DataFrame(train_data, columns=['house_num', 'type', 'length', 'width'])
    # model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")

    side_guess = []
    for i in range(len(guess_list)):
        # inference = BayesianModelSampling(model)
        inference = BayesianModelSampling(model2)
        evidence = [State(var='house_num', state=str(len(guess_list))), State(var='type', state=str(guess_list[i]))]
        data_infer = inference.rejection_sample(evidence=evidence, size=1, return_type='dataframe')
        order = ['house_num', 'type', 'length', 'width']
        data_infer = data_infer[order]
        data_infer = data_infer.values.tolist()
        side_guess.append(dp.str2int(data_infer[0]))

    return side_guess

# 获取训练数据
def get_train_data(info):
    train_data = []
    for i in range(len(info)):
        for j in range(len(info[i])):
            temp_list = []
            temp_list.append(len(info[i]))
            temp_list.append(info[i][j]['label'])
            temp_length = max(info[i][j]['side'][0], info[i][j]['side'][1])
            temp_width = min(info[i][j]['side'][0], info[i][j]['side'][1])
            temp_list.append(temp_length)
            temp_list.append(temp_width)
            train_data.append(temp_list)
    train_data = dis2gaussian(train_data)
    return train_data


# 距离聚类函数
def dis2gaussian(train_data):
    length_gaussian = GaussianMixture(n_components=5, random_state=0)
    width_gaussian = GaussianMixture(n_components=5, random_state=0)
    temp_length = []
    temp_width = []
    for i in range(len(train_data)):
        temp_length.append(train_data[i][2])
        temp_width.append(train_data[i][3])

    temp_length = np.array([temp_length]).reshape(len(temp_length), 1)
    temp_width = np.array([temp_width]).reshape(len(temp_width), 1)
    temp_length = length_gaussian.fit_predict(temp_length)
    temp_width = width_gaussian.fit_predict(temp_width)

    # a = dis_gaussian.means_

    for i in range(len(train_data)):
        train_data[i][2] = temp_length[i]
        train_data[i][3] = temp_width[i]
    return train_data
