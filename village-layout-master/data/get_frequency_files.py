# -*- coding: utf-8 -*-
# @Time : 2020/11/30 20:40
# @Author : zl
# @File : get_frequency_files.py
# @desc:

import csv
import json
import numpy as np
from sklearn.cluster import KMeans
import os
import cv2


def to_array(str):
    """
    str转array坐标函数
    Parameters
    ----------
    str (string)
    """

    # 转成list
    temp = json.loads(str)
    arr = np.array(temp)
    return arr


def delchar(str):
    """
    删除'\'元素
    Parameters
    ----------
    str (string)
    """
    a = str.replace('\'', "\"")
    return a


def str2list(str):
    """
    对房屋中心点坐标'center'操作
    Parameters
    ----------
    str (string)
    """
    l = []
    if (len(str) == 9):
        a = int(str[1:4])
        b = int(str[5:8])
        l.append(a)
        l.append(b)
    else:
        print('转成list失败了')
    return l


def to_dict(str):
    str = delchar(str)
    temp = json.loads(str)
    temp['label'] = int(temp['label'])
    temp['center'] = str2list(temp['center'])
    temp['vercoordinate'] = to_array(temp['vercoordinate']).tolist()
    temp['side'] = to_array(temp['side']).tolist()
    temp['area'] = float(temp['area'])
    temp['angle'] = int(temp['angle'])
    temp['dist_house'] = to_array(temp['dist_house']).tolist()
    temp['dist_road'] = float(temp['dist_road'])
    return temp


def cnts_read_csv(filename):
    """
    从CSV的_block_cnts.csv中读取所有block数据(block房屋类型,house轮廓数据,road数据)
    Parameters
    ----------
    filename (string):如'1',指读取 1_block_cnts.csv 文件
    """

    # 设置文件路径
    CSV_FILE_PATH = '../CSV/' + filename + '_block_cnts.csv'
    # 定义存储数据结构
    data = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            data.append(line)
    # 将所有数据从str转成array
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = to_array(data[i][j])
    return data


def road_read_csv(filename):
    """
    从CSV的_block_cnts.csv中读取所有的road数据,返回所有的block面积
    Parameters
    ----------
    filename (string):如'1'
    """

    # 设置文件路径
    CSV_FILE_PATH = '../CSV/' + filename + '_block_cnts.csv'
    # 定义存储数据结构
    data = []
    road_area = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            data.append(line)
    # 计算block面积(road围成的面积)
    for i in range(len(data)):
        road_area.append(cv2.contourArea(to_array(data[i][-1])))
    return road_area


def info_read_csv(filename):
    """
    从CSV的_block_info.csv中读取每个block中详细房屋数据(label,center,vercoordinate,side,area,angle,dist_house,dist_road)
    Parameters
    ----------
    filename (string):如'1'
    """

    # 设置文件路径
    CSV_FILE_PATH = '../CSV/' + filename + '_block_info.csv'
    # 定义存储数据机构
    data = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            data.append(line)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = to_dict(data[i][j])
    return data


def road_area_divide(cnts_data, road_area, info_data, filename_label):
    """
    根据block面积大小将其划分为5个类别
    Parameters
    ----------
    cnts_data(list):每个block中房屋类别、轮廓信息及道路轮廓信息
    road_area(list):所有的道路包围盒(block)的面积
    info_data(list):每个block中房屋的详细信息(label,center,vercoordinate,side,area,angle,dist_house,dist_road)
    filename_label(string):如：'1' (当前传入的村庄图片序号)
    """

    # 准备划分的block类别数
    num_block_categories = 5
    # 文件存储的目标文件夹
    dest_dir = f"../frequency_files"
    # 如果没有block_categories文件夹则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # 克服每次聚类结果不一致的问题
    # 如果block_categories文件夹已经存在csv文件则不再生成新文件
    else:
        Files = os.listdir(dest_dir)
        for k in range(len(Files)):
            # 提取文件夹内所有文件的后缀
            Files[k] = os.path.splitext(Files[k])[1]
            Str = '.csv'
            if Str in Files:
                print("已经存在.csv文件")
                return

    # 使用KMeans聚类：按照block面积来划分类别
    real_road_area = np.array([road_area]).reshape(len(road_area), 1)
    km = KMeans(n_clusters=num_block_categories).fit(real_road_area)  # 分为5个类别
    label_road_area = (km.fit_predict(real_road_area)).tolist()
    # 将每个类别的block数据分别存入对应类别csv文件
    cnts_block_categories = []
    info_block_categories = []
    for i in range(num_block_categories):
        for j in range(len(label_road_area)):
            if i == label_road_area[j]:
                cnts_temp = cnts_data[j]
                info_temp = info_data[j]
                cnts_temp.insert(0, i)  # 在第一列插入当前block类别标识符
                info_temp.insert(0, i)
                cnts_block_categories.append(cnts_temp)
                info_block_categories.append(info_temp)

    # 先将cnts_block_categories转为list再写入文件存储
    for h in range(len(cnts_block_categories)):
        for m in range(len(cnts_block_categories[h])):
            # 第一列为当前block类别标识符，所以跳过
            if m != 0:
                cnts_block_categories[h][m] = cnts_block_categories[h][m].tolist()

    # 生成两个经block类别划分后的文件
    with open(dest_dir + '/' + filename_label + '_cnts_block_categories.csv', 'w') as file:
        csv_writer = csv.writer(file, lineterminator='\n')
        csv_writer.writerows(cnts_block_categories)
    with open(dest_dir + '/' + filename_label + '_info_block_categories.csv', 'w') as file:
        csv_writer = csv.writer(file, lineterminator='\n')
        csv_writer.writerows(info_block_categories)


def get_house_categories_frequency(filename_label):
    """
    获取按block类别划分后的各个房屋类别出现的频率文件
    Parameters
    ----------
    filename_label(string):如：'1' (当前传入的村庄图片序号)
    """

    dest_dir = f"../frequency_files"
    # 设置文件路径
    CSV_FILE_PATH = '../frequency_files/' + filename_label + '_cnts_block_categories.csv'
    # 定义存储数据结构
    data = []
    # 数据读取
    with open(CSV_FILE_PATH, 'r') as f:
        file = csv.reader(f)
        for line in file:
            data.append(line)
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = to_array(data[i][j])

    # 计算各个block类别中房屋类别频率
    frequency_house_category = []
    for i in range(5):
        merge_house_categories = []
        for j in range(len(data)):
            if data[j][0] == i:
                data[j][1] = data[j][1].tolist()
                merge_house_categories.append(data[j][1])
        # temp存放当前block类别中所有房屋
        temp = []
        for j in range(len(merge_house_categories)):
            for k in range(len(merge_house_categories[j])):
                temp.append(merge_house_categories[j][k])
        temp.sort()
        # num_count(字典)存储当前block类别下各个房屋类别出现的次数
        num_count = {}
        for j in temp:
            if j not in num_count:
                num_count[j] = 1
            else:
                num_count[j] += 1
        # frequency_house_category：记录block类别,房屋类别,该房屋类别出现的次数,计算出现频率
        total = 0
        for j in num_count.keys():
            total = total + num_count[j]
        for j in num_count.keys():
            t = []
            t.append(i)
            t.append(j)
            t.append(num_count[j])
            t.append(float('%0.3f' % (num_count[j] / total)))
            frequency_house_category.append(t)
    # print(frequency_house_category)
    with open(dest_dir + '/' + filename_label + '_house_categories_frequency.csv', 'w') as file:
        csv_writer = csv.writer(file, lineterminator='\n')
        csv_writer.writerows(frequency_house_category)


def get_house_area_proportion(filename_label):
    """
    获取选择的block类别的 各类别房屋平均尺寸与该类别block尺寸占比率 文件
    Parameters
    ----------
    filename_label(string):如：'1' (当前传入的村庄图片序号)
    """

    dest_dir = f"../frequency_files"

    #################  获取当前村庄图片中各个block类别的平均面积  #################
    # 设置文件路径
    CSV_FILE_PATH_1 = '../frequency_files/' + filename_label + '_cnts_block_categories.csv'
    # 定义存储数据结构
    cnts_data = []
    road_area = []
    # 数据读取 road_area：记录block类别，block面积
    with open(CSV_FILE_PATH_1, 'r') as f:
        file = csv.reader(f)
        for line in file:
            cnts_data.append(line)
    for i in range(len(cnts_data)):
        t = []
        t.append(cnts_data[i][0])
        t.append(cv2.contourArea(to_array(cnts_data[i][-1])))
        road_area.append(t)
    # print(road_area)  # 测试

    r = {}  # 每个block类别的面积集合
    for i in road_area:
        m = str(i[0])
        if (m in r.keys()):
            r[m].append(i[1])
        else:
            r[m] = []
            r[m].append(i[1])
    # print(r)  # 测试

    # 计算每个block类别的平均面积
    block_average_area = []
    for i in r.keys():
        t = []
        t.append(int(i[0]))
        t.append((sum(r[i]) / len(r[i])))
        block_average_area.append(t)
    # print(block_average_area)  # 测试

    #################  获取当前村庄图片中各个block类别中各房屋类别的平均面积  #################
    # 设置文件路径
    CSV_FILE_PATH_2 = '../frequency_files/' + filename_label +'_info_block_categories.csv'
    # 定义存储数据结构
    info_data = []
    # 数据读取
    with open(CSV_FILE_PATH_2, 'r') as f:
        file = csv.reader(f)
        for line in file:
            info_data.append(line)
    for i in range(len(info_data)):
        for j in range(len(info_data[i])):
            if j != 0:
                str1 = delchar(info_data[i][j])
                info_data[i][j] = json.loads(str1)

    # all_house_area : 记录block类别,房屋类别,房屋面积
    all_house_area = []
    for i in range(5):
        for j in range(len(info_data)):
            if int(info_data[j][0]) == i:
                for m in range(len(info_data[j])):
                    t = []
                    for k in range(10):
                        if m != 0 and info_data[j][m].get('label') == k:
                            t.append(i)
                            t.append(info_data[j][m].get('label'))
                            t.append(info_data[j][m].get('area'))
                            all_house_area.append(t)
    # print(all_house_area)  # 测试

    temp = {}  # 每个block类别及里面房屋类别的面积集合:如'04':[]
    for i in all_house_area:
        mid = str(i[0])
        mid += str(i[1])
        # print(mid)  # 测试
        if(mid in temp.keys()):
            temp[mid].append(i[2])
        else:
            temp[mid] = []
            temp[mid].append(i[2])
    # print(temp)  # 测试

    # 计算block类别中各个房屋类别平均面积
    house_average_area = []
    for i in temp.keys():
        t = []
        t.append(int(i[0]))
        t.append(int(i[1]))
        t.append((sum(temp[i])/len(temp[i])))
        house_average_area.append(t)
    # print(house_average_area)  # 测试

    #################  计算该block类别中尺寸占比： 某房屋类别平均面积/该block类别平均面积   #################
    for i in range(len(block_average_area)):
        for j in range(len(house_average_area)):
            if house_average_area[j][0] == block_average_area[i][0]:
                house_average_area[j][2] = float('%0.3f' % float(house_average_area[j][2]/block_average_area[i][1]))

    with open(dest_dir + '/' + filename_label + '_house_area_proportion.csv', 'w') as file:
        csv_writer = csv.writer(file, lineterminator='\n')
        csv_writer.writerows(house_average_area)



if __name__ == '__main__':
    # 测试
    a = cnts_read_csv('1')
    b = road_read_csv('1')
    c = info_read_csv('1')
    road_area_divide(a, b, c, '1')

    get_house_categories_frequency('1')
    get_house_area_proportion('1')
