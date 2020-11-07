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

info = dp.info_read_csv('1')
road_area = []
house_num = []
house_type = []
data = dp.cnts_read_csv('1')

for i in range(len(data)):
    road_area.append(cv2.contourArea(data[i][-1]))
    house_num.append(len(data[i][0]))
    house_type.append(data[i][0])

model = BayesianModel([('block_area', 'house_num'), ('block_area', 'type_0'),
                       ('block_area', 'type_1'), ('block_area', 'type_2'), ('block_area', 'type_3'),
                       ('block_area', 'type_5'), ('block_area', 'type_6'), ('block_area', 'type_7'),
                       ('block_area', 'type_8'), ('block_area', 'type_9'), ('house_num', 'type_0'),
                       ('house_num', 'type_1'), ('house_num', 'type_2'), ('house_num', 'type_3'),
                       ('house_num', 'type_4'),
                       ('house_num', 'type_5'), ('house_num', 'type_6'), ('house_num', 'type_7'),
                       ('house_num', 'type_8'), ('house_num', 'type_9'), ('type_0', 'num_0'),
                       ('type_1', 'num_1'), ('type_2', 'num_2'), ('type_3', 'num_3'),
                       ('type_4', 'num_4'), ('type_5', 'num_5'), ('type_6', 'num_6'),
                       ('type_7', 'num_7'), ('type_8', 'num_8'), ('type_9', 'num_9')])

road_area = np.array([road_area]).reshape(len(road_area), 1)
km = KMeans(n_clusters=10).fit(road_area)  # 将数据集分为3类
road_area = km.fit_predict(road_area)

#
all = []
for i in range(len(data)):
    item = []
    item.append(road_area[i])
    item.append(len(house_type[i]))
    for k in range(20):
        item.append(int(0))
    for j in range(len(data[i][0])):
        # house = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # a = data[i][0][j]
        item[data[i][0][j] + 2] = 1
        item[data[i][0][j] + 12] = item[data[i][0][j] + 12] + 1
    all.append(item)

df = pd.DataFrame(all, columns=['block_area', 'house_num','type_0','type_1','type_2','type_3','type_4','type_5'
                                ,'type_6','type_7','type_8','type_9','num_0','num_1','num_2','num_3','num_4',
                                'num_5','num_6','num_7','num_8','num_9'])
model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")
for cpd in model.get_cpds():
    print(cpd)
# dp.showBN(model)
print(df)
