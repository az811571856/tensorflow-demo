import pandas as pd
# 读取数据
df_train = pd.read_csv("train.csv", index_col='user_id')
df_valid = pd.read_csv("test.csv", index_col='user_id')

# 处理数据 1、删除无用列 2、处理分类label为数字 3、把类别编码 4、分割训练集为训练和测试
df_train.drop(['EmployeeNumber'], axis=1)
df_valid.drop(['EmployeeNumber'], axis=1)
df_train["Attrition"] = df_train["Attrition"].map(lambda x: (0 if x == "No" else 1))
cate_list = ['Age', 'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
from sklearn.preprocessing import LabelEncoder
for cate in cate_list:
    labelEncoder = LabelEncoder()
    df_train[cate] = labelEncoder.fit_transform(df_train[cate])
    df_valid[cate] = labelEncoder.transform(df_valid[cate])
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_train.drop('Attrition', axis=1), df_train['Attrition'], test_size=0.2, random_state=42)
# 输入模型
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(verbose=True, random_state=33)
# 训练
lr.fit(train_x, train_y)


result =pd.DataFrame(columns=('Attrition1','Attrition'))


# 预测
result['Attrition1'] = lr.predict(df_valid)
result['Attrition2']=lr.predict_proba(df_valid)[:, 1]
result['Attrition']=result['Attrition2'].map(lambda x:1 if x>=0.5 else 0)
# 保存结果
result[['Attrition','Attrition1','Attrition2']].to_csv('result1.csv')