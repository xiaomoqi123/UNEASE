#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys

import numpy as np
import pandas as pd
import sklearn
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from joblib import Memory
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from minepy import MINE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from LR import LR
from sklearn.ensemble import GradientBoostingClassifier
#from stability_selection.randomized_lasso import RandomizedLogisticRegression

'方差选择法'
def variance_select_fea(fea, k):
    col = list(fea.columns)
    # 方差选择法，返回值为特征选择后的数据
    # 参数threshold为方差的阈值
    selector = VarianceThreshold(threshold=0)
    selector.fit_transform(fea)
    var = selector.variances_
    '升序'
    var = sorted(var, reverse=True)
    t = var[k]
    selector = VarianceThreshold(threshold=t)
    fea = selector.fit_transform(fea)
    # print(fea)
    # print(selector.get_support(indices=True))
    index = selector.get_support(indices=True)
    col_names = [col[i] for i in index]
    df = pd.DataFrame(data=fea, columns=col_names)
    return df

'相关系数法，需要标签'
def correlate_select(fea, label_dic, k):
    col = list(fea.columns)
    # 选择K个最好的特征，返回选择特征后的数据
    # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
    # 输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
    # 参数k为选择的特征个数
    label_list = list(label_dic.values())
    selector = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T, k=k)
    fea = selector.fit_transform(fea, label_list)
    index = selector.get_support(indices=True)
    col_names = [col[i] for i in index]
    df = pd.DataFrame(data=fea, columns=col_names)
    return df

'卡方检验,需要标签'
def chi2_select(fea, label_dic, k):
    col = list(fea.columns)
    label_list = list(label_dic.values())
    # 选择K个最好的特征，返回选择特征后的数据
    '输入fea必须非负'
    fea = MinMaxScaler().fit_transform(fea)
    selector = SelectKBest(chi2, k=k)
    fea = selector.fit_transform(fea, label_list)
    index = selector.get_support(indices=True)
    col_names = [col[i] for i in index]
    df = pd.DataFrame(data=fea, columns=col_names)
    return df

'互信息法'
#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)

def mutual_information(fea, label_dic, k):
    col = list(fea.columns)
    label_list = list(label_dic.values())
    # 选择K个最好的特征，返回选择特征后的数据
    selector = SelectKBest(lambda X, Y: np.array(list(map(lambda x: mic(x, Y)[0], X.T))).T, k=k)
    fea = selector.fit_transform(fea, label_list)
    index = selector.get_support(indices=True)
    col_names = [col[i] for i in index]
    df = pd.DataFrame(data=fea, columns=col_names)
    return df

'递归特征消除法'
def recur_elimination(fea, label_dic, k):
    col = list(fea.columns)
    label_list = list(label_dic.values())
    #递归特征消除法，返回特征选择后的数据
    #参数estimator为基模型
    #参数n_features_to_select为选择的特征个数
    selector = RFE(estimator=LogisticRegression(), n_features_to_select=k)
    fea = selector.fit_transform(fea, label_list)
    index = selector.get_support(indices=True)
    col_names = [col[i] for i in index]
    df = pd.DataFrame(data=fea, columns=col_names)
    return df

'基于惩罚项的特征选择法——l1'
def Penalty_l1(fea, label_dic):
    col = list(fea.columns)
    label_list = list(label_dic.values())
    #带L1惩罚项的逻辑回归作为基模型的特征选择
    #l1必须要——solver='liblinear'
    selector = SelectFromModel(LogisticRegression(penalty="l1", C=0.1, solver='liblinear'))
    fea = selector.fit_transform(fea, label_list)
    index = selector.get_support(indices=True)
    col_names = [col[i] for i in index]
    df = pd.DataFrame(data=fea, columns=col_names)
    return df

'基于惩罚项的特征选择法——l1 l2'
def Penalty_l1l2(fea, label_dic):
    col = list(fea.columns)
    label_list = list(label_dic.values())
    # 带L1和L2惩罚项的逻辑回归作为基模型的特征选择
    # 参数threshold为权值系数之差的阈值
    selector = SelectFromModel(LR(threshold=0.5, C=0.1))
    fea = selector.fit_transform(fea, label_list)
    index = selector.get_support(indices=True)
    col_names = [col[i] for i in index]
    df = pd.DataFrame(data=fea, columns=col_names)
    return df

'基于树模型GBDT'
def GBDT(fea, label_dic):
    col = list(fea.columns)
    label_list = list(label_dic.values())
    # GBDT作为基模型的特征选择
    selector = SelectFromModel(GradientBoostingClassifier())
    fea = selector.fit_transform(fea, label_list)
    index = selector.get_support(indices=True)
    col_names = [col[i] for i in index]
    df = pd.DataFrame(data=fea, columns=col_names)
    return df
