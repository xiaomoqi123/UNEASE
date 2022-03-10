#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy
import pandas as pd

class DBScan:

    def __init__(self, data_fea, eps, min_Pts):
        self.fea = data_fea

        fea = pd.DataFrame(self.fea.values.T, index=self.fea.columns, columns=self.fea.index)
        # fea每行是一个特征的所有值，每列是一个样本
        # eps = eps
        # min_Pts = min_Pts
        fea_np = np.array(fea)
        C = self.DBSCAN(fea_np, eps, min_Pts)
        self.clusterData = C
        # print(C)

    def find_neighbor(self, j, x, eps):
        N = list()
        temp = np.sum((x - x[j]) ** 2, axis=1) ** 0.5
        #print(temp)
        # print(np.argwhere(temp <= eps))
        N = np.argwhere(temp <= eps).flatten().tolist()
        return set(N)


    def DBSCAN(self, X, eps, min_Pts):
        k = -1
        neighbor_list = []  # 用来保存每个数据的邻域
        omega_list = []  # 核心对象集合
        gama = set([x for x in range(len(X))])  # 初始时将所有点标记为未访问
        cluster = [-1 for _ in range(len(X))]  # 聚类
        for i in range(len(X)):
            neighbor_list.append(self.find_neighbor(i, X, eps))
            if len(neighbor_list[-1]) >= min_Pts:
                omega_list.append(i)  # 将样本加入核心对象集合
        omega_list = set(omega_list)  # 转化为集合便于操作
        while len(omega_list) > 0:
            gama_old = copy.deepcopy(gama)
            j = random.choice(list(omega_list))  # 随机选取一个核心对象
            k = k + 1
            Q = list()
            Q.append(j)
            gama.remove(j)
            while len(Q) > 0:
                q = Q[0]
                Q.remove(q)
                if len(neighbor_list[q]) >= min_Pts:
                    delta = neighbor_list[q] & gama
                    deltalist = list(delta)
                    for i in range(len(delta)):
                        Q.append(deltalist[i])
                        gama = gama - delta
            Ck = gama_old - gama
            Cklist = list(Ck)
            for i in range(len(Ck)):
                cluster[Cklist[i]] = k
            omega_list = omega_list - Ck
        cluster = np.array(cluster)
        return cluster

