#!/usr/bin/python
# -*- coding: utf-8 -*-
import random

import numpy as np

class Obstacle4:
    def __init__(self):
        self.obstacle = np.array([[2, 5, 2],
                                  [4, 3, 2],
                                  [8, 8, 2]
                                  ], dtype=float)  # 球障碍物坐标
        # self.cylinder = np.array([[4, 7],[2, 3],[5, 7],[8, 7],[9, 11],[10, 13],[11, 15],[12, 7],[13, 3],[15, 5],[18, 10],
        #                           [12, 18],[14, 18],[16, 18],[19,17]
        #                           ], dtype=float)  # 圆柱体障碍物坐标（只需要给定圆形的x,y即可，无顶盖）
        # self.cylinderR = np.array([[1],[1],[1],[1],[2],[1],[1],[1],[2],[1],[1],[3],[2],[3],[2]], dtype=float)  # 圆柱体障碍物半径
        self.cylinderH = np.array([[0.5],[0.7],[1.0],[0.5],[0.5],[0.7],[1.0],[0.5],[0.5],[0.7],[1.0],[0.5],[0.3],[0.2],[0.5]], dtype=float)  # 圆柱体高度

        ob = [[random.randint(0, 10) for j in range(1, 3)] for i in range(1, 50)]
        for ob_info in ob:
            if ob_info == [10,10] or ob_info == [9,10] or ob_info == [10,9] or ob_info == [9,9] or ob_info == [2,8] : #
                ob.pop(ob.index(ob_info))

        self.cylinder = np.array(ob)
        ob_r_list = []
        for i in range(len(ob)):
            ob_r_list.append(1)
        self.cylinderR = np.array(ob_r_list)

        self.qgoal = np.array([10, 10, 0], dtype=float)  # 目标点
        self.x0 = np.array([0, 0, 0], dtype=float)  # 轨迹起始点

seed = 3
random.seed(seed)

Obstacle = {"Obstacle4":Obstacle4()}