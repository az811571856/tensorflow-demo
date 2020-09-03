# coding:utf-8
#https://www.zybuluo.com/rianusr/note/1195225
# 数据分析/人工智能经典案例04 - 基于SVD协同过滤算法实现的电影推荐系统
import numpy as np
import pandas as pd
# 导入数据
data=pd.read_csv('ml-100k/u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])
# 用户物品统计
n_users = data.user_id.nunique()
n_items = data.item_id.nunique()
# 拆分数据集
from sklearn.model_selection import train_test_split
train_data,test_data =train_test_split(data,test_size=0.3) #按照训练集70%，测试集30%的比例对数据进行拆分
# 训练集 用户-物品 矩阵
user_item_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    user_item_matrix[line[1]-1,line[2]-1] = line[3]
# 构建物品相似矩阵 - 使用sklearn.metrics.pairwise中的cosine计算余弦距离
'''
采用余弦距离计算相似度
如果两个物品在同一条水平线上，则其夹角为零，对应的余弦值为1，代表完全相似
如果两个物品处于垂直的方向上，其夹角为90度，那么其余弦值为0，代表毫不相干
'''
from sklearn.metrics.pairwise import pairwise_distances
# 相似度计算定义为余弦距离
item_similarity_m = pairwise_distances(user_item_matrix.T,metric='cosine')
# 物品相似矩阵探索
'''
item_similarity_m.shape     >> (1682, 1682)
item_similarity_m[0:5,0:5].round(2) # 取5*5的矩阵查看其保留两位小数的数据
# pairwise_distances模块在计算物品相似性时，不会计算自己与自己的相似性，所以所以对角线的值都为0
>> array([[0.  , 0.67, 0.73, 0.7 , 0.81],
       [0.67, 0.  , 0.84, 0.64, 0.82],
       [0.73, 0.84, 0.  , 0.8 , 0.85],
       [0.7 , 0.64, 0.8 , 0.  , 0.76],
       [0.81, 0.82, 0.85, 0.76, 0.  ]])
'''
# 现在我们只分析上三角，得到等分位数
item_similarity_m_triu = np.triu(item_similarity_m,k=1) # 取得上三角数据
item_sim_nonzero = np.round(item_similarity_m_triu[item_similarity_m_triu.nonzero()],3)
'''
# 上三角矩阵
arr=np.linspace(1,9,9).reshape(3,3)
arr
>> array([[1., 2., 3.],
       [4., 5., 6.],
       [7., 8., 9.]])
np.triu(arr,k=1) # 默认k=0，k的值正数表示向右上角移对应个单位，把对应位置全部变为0
>> array([[0., 2., 3.],
       [0., 0., 6.],
       [0., 0., 0.]])
'''
# 查看十分位数
np.percentile(item_sim_nonzero,np.arange(0,101,10))
user_item_precdiction_non_stan = user_item_matrix.dot(item_similarity_m)
user_item_precdiction = user_item_precdiction_non_stan / np.array([np.abs(item_similarity_m).sum(axis=1)])
# 除以np.array([np.abs(item_similarity_m).sum(axis=1)]是为了可以使评分在1~5之间，使1~5的标准化
print(user_item_precdiction_non_stan[:1])
print(user_item_precdiction[:1])

# 只取数据集中有评分的数据集进行评估
from sklearn.metrics import mean_squared_error
from math import sqrt
train_item_matrix = user_item_matrix
prediction_flatten = user_item_precdiction[train_item_matrix.nonzero()]
user_item_matrix_flatten = train_item_matrix[train_item_matrix.nonzero()]
error_train = sqrt(mean_squared_error(prediction_flatten,user_item_matrix_flatten))  # 均方根误差计算
print('训练集预测均方根误差：', error_train)

test_data_matrix = np.zeros((n_users,n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1,line[2]-1]=line[3]
# 预测矩阵
item_prediction = test_data_matrix.dot(item_similarity_m) / np.array(np.abs(item_similarity_m).sum(axis=1))
# 只取数据集中有评分的数据集进行评估
prediction_flatten = user_item_precdiction[test_data_matrix.nonzero()]
test_data_matrix_flatten = test_data_matrix[test_data_matrix.nonzero()]
error_test = sqrt(mean_squared_error(prediction_flatten,test_data_matrix_flatten))  # 均方根误差计算
print('测试集预测均方根误差：', error_test)

