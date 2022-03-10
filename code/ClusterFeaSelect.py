#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial import distance_matrix
import math
from KMeans import Kmeans
from DBscan import DBScan
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV

def std_method(zscore_fea, clusterData):
    # 计算每个特征向量的标准差
    std_value = np.array(np.zeros(zscore_fea.shape[1]))
    for i in range(zscore_fea.shape[1]):
        std_value[i] = np.std(zscore_fea[zscore_fea.columns[i]], ddof=1)
    # 选择最具代表性特征
    std_value = pd.DataFrame(std_value)
    std_value = pd.DataFrame(std_value.values.T, index=['std'], columns=zscore_fea.columns)
    new_df = pd.concat([zscore_fea, std_value])
    clusterData = pd.DataFrame(clusterData)
    clusterData = pd.DataFrame(clusterData.values.T, index=['class'], columns=zscore_fea.columns)
    new_df = pd.concat([new_df, clusterData])
    new_df = pd.DataFrame(new_df.values.T, index=new_df.columns, columns=new_df.index)
    new_df = new_df.groupby(['class'])['std'].idxmax()
    # print("=====selected feature============")
    # print(zscore_fea[new_df])
    return zscore_fea[new_df]

def inclass_corr_method(zscore_fea, clusterData):
    # 计算每个特征向量的类内相关性
    corr_value = np.array(np.zeros(zscore_fea.shape[1]))
    #clusterData = pd.DataFrame(obj.clusterData)
    for i in range(zscore_fea.shape[1]):
        corr = 0
        list = np.where(clusterData==clusterData[i])
        list = list[0]
        for j in range(len(list)):
            if i != list[j]:
                pea = pearsonr(zscore_fea.iloc[:, i], zscore_fea.iloc[:, list[j]])
                corr += 0 if np.isnan(pea[0]) else pea[0]
            else:
                corr += 0
        corr_value[i] = corr

    # 选择最具代表性特征
    std_value = pd.DataFrame(corr_value)
    std_value = pd.DataFrame(std_value.values.T, index=['corr'], columns=zscore_fea.columns)
    new_df = pd.concat([zscore_fea, std_value])
    clusterData = pd.DataFrame(clusterData)
    clusterData = pd.DataFrame(clusterData.values.T, index=['class'], columns=zscore_fea.columns)
    new_df = pd.concat([new_df, clusterData])
    new_df = pd.DataFrame(new_df.values.T, index=new_df.columns, columns=new_df.index)
    new_df = new_df.groupby(['class'])['corr'].idxmax()
    # print("=====selected feature============")
    # print(zscore_fea[new_df])
    return zscore_fea[new_df]

def LDF_method(zscore_fea, clusterData, r):
    # 计算每个特征向量的半径r内的密度
    corr_value = np.array(np.zeros(zscore_fea.shape[1]))
    for i in range(zscore_fea.shape[1]):
        corr = 0
        list = np.where(clusterData == clusterData[i])
        list = list[0]
        for j in range(len(list)):
            if i != list[j]:
                eu = np.sqrt(sum((zscore_fea.iloc[:, i] - zscore_fea.iloc[:, list[j]]) ** 2))
                corr += 1 if eu < r else 0
            else:
                continue
        corr_value[i] = corr

    # 选择最具代表性特征
    std_value = pd.DataFrame(corr_value)
    std_value = pd.DataFrame(std_value.values.T, index=['corr'], columns=zscore_fea.columns)
    new_df = pd.concat([zscore_fea, std_value])
    clusterData = pd.DataFrame(clusterData)
    clusterData = pd.DataFrame(clusterData.values.T, index=['class'], columns=zscore_fea.columns)
    new_df = pd.concat([new_df, clusterData])
    new_df = pd.DataFrame(new_df.values.T, index=new_df.columns, columns=new_df.index)
    new_df = new_df.groupby(['class'])['corr'].idxmax()
    # print("=====selected feature============")
    # print(zscore_fea[new_df])
    return zscore_fea[new_df]

def calculate_entropy(df):
    df = df.transpose()
    #每行间的欧氏距离矩阵
    dis_matrix = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
    a=0.5
    E = 0
    for i in range(dis_matrix.shape[1]):
        for j in range(dis_matrix.shape[1]):
            if i == j:
                continue
            sij = math.exp(-a * dis_matrix.iloc[i, j])
            #sij=1会出错，因为log里面不能是0
            E += -(sij * math.log(sij) + (1-sij) * math.log(1-sij)) if 1-sij != 0 else 0
    return E

def entropy_method(zscore_fea, clusterData):
    # 计算每个聚类中，去掉自己后聚类的熵，最小的就是代表这个聚类的
    corr_value = np.array(np.zeros(zscore_fea.shape[1]))
    for i in range(zscore_fea.shape[1]):
        list = np.where(clusterData == clusterData[i])
        new_df = zscore_fea.iloc[:, list[0]]
        del new_df[zscore_fea.columns[i]]
        corr_value[i] = calculate_entropy(new_df)

    # 选择最具代表性特征
    std_value = pd.DataFrame(corr_value)
    std_value = pd.DataFrame(std_value.values.T, index=['corr'], columns=zscore_fea.columns)
    new_df = pd.concat([zscore_fea, std_value])
    clusterData = pd.DataFrame(clusterData)
    clusterData = pd.DataFrame(clusterData.values.T, index=['class'], columns=zscore_fea.columns)
    new_df = pd.concat([new_df, clusterData])
    new_df = pd.DataFrame(new_df.values.T, index=new_df.columns, columns=new_df.index)
    #选择熵最小的
    new_df = new_df.groupby(['class'])['corr'].idxmin()
    # print("=====selected feature============")
    # print(zscore_fea[new_df])
    return zscore_fea[new_df]

def cluster_select_fea(zscore_fea, args):
    '''
    #预处理
    values = fea.values  # dataframe转换为array
    values = values.astype('float32')  # 定义数据类型
    data = preprocessing.scale(values)
    zscore_fea = pd.DataFrame(data)  # 将array还原为dataframe
    zscore_fea.columns = fea.columns  # 命名标题行
    '''

    #聚类
    if args.feaSelect == "k-means":
        obj = Kmeans(correlate=args.correlate, data_fea=zscore_fea)
        clusterData = obj.clusterData
    elif args.feaSelect == "DBSCAN":

        obj = DBScan(data_fea=zscore_fea, eps=args.eps, min_Pts=args.min_Pts)
        clusterData = obj.clusterData
        '''
        'python的DBSCAN库'
        fea = pd.DataFrame(zscore_fea.values.T, index=zscore_fea.columns, columns=zscore_fea.index)
        # 设置半径为args.eps，最小样本量为args.min_Pts，建模
        db = DBSCAN(eps=args.eps, min_samples=args.min_Pts).fit(fea)
        labels = db.labels_
        clusterData = labels
        '''


        # 注：cluster列是kmeans聚成3类的结果；cluster2列是kmeans聚类成2类的结果；
        # scaled_cluster列是kmeans聚类成3类的结果（经过了数据标准化）
    #k = np.unique(obj.clusterData)

    '如果只有一个聚类，一个聚类选择一个特征值，那么在选择训练集的时候，很有可能会将数据集都划分为训练集，因此在划分训练集时修改一下'

    if args.select_method == "std":
        new_fea = std_method(zscore_fea, clusterData)
    elif args.select_method == "inclass_corr":
        new_fea = inclass_corr_method(zscore_fea, clusterData)
    elif args.select_method == "LDF":
        new_fea = LDF_method(zscore_fea, clusterData, args.LDF_r)
    elif args.select_method == "entropy":
        new_fea = entropy_method(zscore_fea, clusterData)
    return new_fea
