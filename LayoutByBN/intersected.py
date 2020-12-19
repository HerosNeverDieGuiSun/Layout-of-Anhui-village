# -*- coding: utf-8 -*- 
# @Time : 2020/12/19 16:39 
# @Author : zzd 
# @File : intersected.py 
# @desc:

# 点
class Point(object):
    def __init__(self, x, y):
        self.x, self.y = x, y


# 向量
class Vector(object):
    def __init__(self, start_point, end_point):
        self.start, self.end = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y
#
def negative(vector):
    return Vector(vector.end,vector.start)

def vertor_product(vector1, vector2):
    return vector1.x * vector2.y - vector2.x * vector1.y

def is_intersected(A,B,C,D):
    A = Point(A[0],A[1])
    B = Point(B[0], B[1])
    C = Point(C[0], C[1])
    D = Point(D[0], D[1])
    AC = Vector(A,C)
    AD = Vector(A,D)
    BC = Vector(B,C)
    BD = Vector(B,D)
    CA = negative(AC)
    CB = negative(BC)
    DA = negative(AD)
    DB = negative(BD)
    ZERO = 1e-9
    if (vertor_product(AC,AD) * vertor_product(BC,BD) <= ZERO)  and (vertor_product(CA,CB) * vertor_product(DA,DB) <= ZERO):
        return [[C.x,C.y],[D.x,D.y]]
    return 0


