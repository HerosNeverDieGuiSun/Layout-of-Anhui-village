# -*- coding: utf-8 -*- 
# @Time : 2021/1/17 9:39 
# @Author : zzd 
# @File : relation_graph.py 
# @desc: 定义结点的关系图

class Relation_node:

    def __init__(self, key, guess_index, selected):
        self.key = key
        self.north = None
        self.south = None
        self.east = None
        self.west = None
        self.guess_index = guess_index
        self.selected = selected

    def add_toward(self, relation_node, toward):
        if relation_node.selected == 0:
            if toward == 'N' and self.north == None:
                self.north = relation_node
                relation_node.south = self
                return 1
            elif toward == 'S' and self.south == None:
                self.south = relation_node
                relation_node.north = self
                return 1
            elif toward == 'E' and self.east == None:
                self.east = relation_node
                relation_node.west = self
                return 1
            elif toward == 'W' and self.west == None:
                self.west = relation_node
                relation_node.east = self
                return 1
            else:
                return 0
        else:
            return 0

    def change_selected(self, selected):
        self.selected = selected

    def change_pos(self,pos):
        self.pos = pos
