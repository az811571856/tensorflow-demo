# coding:utf-8
# 导入数据
import numpy as np
import pandas as pd
data=pd.read_csv('ml-100k/u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])
# 用户物品统计
n_users = data.user_id.nunique()
n_items = data.item_id.nunique()
# 拆分数据集
from sklearn.model_selection import train_test_split
# 按照训练集70%，测试集30%的比例对数据进行拆分
train_data,test_data =train_test_split(data,test_size=0.3)
# 训练集 用户-物品 矩阵
user_item_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    user_item_matrix[line[1]-1,line[2]-1] = line[3]
# 构建用户相似矩阵 - 采用余弦距离
from sklearn.metrics.pairwise import pairwise_distances
# 相似度计算定义为余弦距离
user_similarity_m = pairwise_distances(user_item_matrix,metric='cosine') # 每个用户数据为一行，此处不需要再进行转置
user_similarity_m[0:5,0:5].round(2) # 取5*5的矩阵查看其保留两位小数的数据
'''
>> array([[0.  , 0.85, 0.96, 0.96, 0.74],
       [0.85, 0.  , 0.99, 0.84, 0.93],
       [0.96, 0.99, 0.  , 0.77, 0.97],
       [0.96, 0.84, 0.77, 0.  , 0.97],
       [0.74, 0.93, 0.97, 0.97, 0.  ]])
'''
# 现在我们只分析上三角，得到等分位数
user_similarity_m_triu = np.triu(user_similarity_m,k=1) # 取得上三角数据
user_sim_nonzero = np.round(user_similarity_m_triu[user_similarity_m_triu.nonzero()],3)
np.percentile(user_sim_nonzero,np.arange(0,101,10))

mean_user_rating  = user_item_matrix.mean(axis=1)
rating_diff = (user_item_matrix - mean_user_rating[:,np.newaxis]) # np.newaxis作用：为mean_user_rating增加一个维度，实现加减操作
user_precdiction = mean_user_rating[:,np.newaxis] + user_similarity_m.dot(rating_diff) / np.array([np.abs(user_similarity_m).sum(axis=1)]).T
# 处以np.array([np.abs(user_similarity_m).sum(axis=1)]是为了可以使评分在1~5之间，使1~5的标准化
# 只取数据集中有评分的数据集进行评估
from sklearn.metrics import mean_squared_error
from math import sqrt
prediction_flatten = user_precdiction[user_item_matrix.nonzero()]
user_item_matrix_flatten = user_item_matrix[user_item_matrix.nonzero()]
error_train = sqrt(mean_squared_error(prediction_flatten,user_item_matrix_flatten))  # 均方根误差计算
print('训练集预测均方根误差：', error_train)

test_data_matrix = np.zeros((n_users,n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1,line[2]-1]=line[3]
# 预测矩阵
rating_diff = (test_data_matrix - mean_user_rating[:,np.newaxis]) # np.newaxis作用：为mean_user_rating增加一个维度，实现加减操作
user_precdiction = mean_user_rating[:,np.newaxis] + user_similarity_m.dot(rating_diff) / np.array([np.abs(user_similarity_m).sum(axis=1)]).T
# 只取数据集中有评分的数据集进行评估
prediction_flatten = user_precdiction[user_item_matrix.nonzero()]
user_item_matrix_flatten = user_item_matrix[user_item_matrix.nonzero()]
error_test = sqrt(mean_squared_error(prediction_flatten,user_item_matrix_flatten))  # 均方根误差计算
print('测试集预测均方根误差：', error_test)