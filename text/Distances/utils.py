# -*- coding: utf-8 -*-
# @Time : 2020/10/14 21:15
# @Author : zl
# @File : utils.py
# @desc:  KD树的构建相关文件
from typing import List


def get_eu_dist(arr1: List, arr2: List) -> float:
    """Calculate the Euclidean distance of two vectors.
    Arguments:
        arr1 {list} -- 1d list object with int or float
        arr2 {list} -- 1d list object with int or float
    Returns:
        float -- Euclidean distance
    """

    return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5