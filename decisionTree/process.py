# https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sbn


def get_title(name):
    title_search = re.search(r'([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''


def do_precess(data_set):
    '''
    数据预处理的函数，主要是：
    1. nan补全 Embarked Fare Age
    2. 连续值离散化 Age Fare
    3. 字符串数值化 Has_Cabin Title Sex Embarked
    4. 生成派生属性 IsAlone FamilySize
    5. 删除多余属性 PassengerId Name'Ticket Cabin'SibSp
    :param data_set:
    :return:
    '''
    # Cabin
    data_set['Has_Cabin'] = data_set['Cabin']. apply(
        lambda x: 0 if type(x)==float else 1)
    # FamilySize derive
    data_set['FamilySize'] = data_set['SibSp'] + data_set['Parch'] + 1
    # FamilySize  IsAlone
    data_set['IsAlone'] = 0
    data_set.loc[data_set['FamilySize'] == 1, 'IsAlone'] = 1
    # Embarked
    data_set['Embarked'] = data_set['Embarked'].fillna('S')
    data_set['Embarked'] = data_set['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    # Fare
    data_set['Fare'] = data_set['Fare'].fillna(data_set['Fare'].median())
    data_set.loc[data_set['Fare'] <= 7.91, 'Fare'] = 0
    data_set.loc[(data_set['Fare'] > 7.91) & (data_set['Fare'] <= 14.454), 'Fare'] = 1
    data_set.loc[(data_set['Fare'] > 14.454) & (data_set['Fare'] <= 31), 'Fare'] = 2
    data_set.loc[data_set['Fare'] > 31, 'Fare'] = 3
    data_set['Fare'] = data_set['Fare'].astype(int)
    #  Age
    age_avg = data_set['Age'].mean()
    age_std = data_set['Age'].std()
    age_null_count = data_set['Age'].isnull().sum()
    age_null_random_list = np.random.randint(
        age_avg - age_std, age_avg+age_std, size=age_null_count)
    data_set.loc[np.isnan(data_set['Age']), 'Age'] = age_null_random_list
    data_set['Age'] = data_set['Age'].apply(lambda x: int(x / 6)).astype(int)
    # Title
    data_set['Title'] = data_set['Name'].apply(get_title)
    data_set['Title'] = data_set['Title'].replace(
        ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data_set['Title'] = data_set['Title'].replace('Mlle', 'Miss')
    data_set['Title'] = data_set['Title'].replace('Ms', 'Miss')
    data_set['Title'] = data_set['Title'].replace('Mme', 'Mrs')
    data_set['Title'] = data_set['Title'].map(
        {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5})
    data_set['Title'] = data_set['Title'].fillna(0)
    #  Sex
    data_set['Sex'] = data_set['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # Drop
    return data_set.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp'], axis=1)



if __name__ =='__main__':
    train = pd.read_csv('../data/titanic/train.csv')
    test = pd.read_csv('../data/titanic/test.csv')

    train = do_precess(train)
    test = do_precess(test)
    # 查看协方差，观察各个数据的相关系数
    corr = train.astype(float).corr()
    plt.figure(figsize=(12,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sbn.heatmap(corr, linewidths=0.1,vmax=1.0, square=True, linecolor='white', annot=True)
    plt.show()