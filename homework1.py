# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:00:51 2020

@author: user
"""

import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import graphviz
#%%資料來源
train_dataset = 'adult.data'
test_dataset = 'adult.test'

reader = csv.reader(open(train_dataset))
test_reader = csv.reader(open(test_dataset))

#%%

age = []
workclass = []
fnlwgt = []
education = []
education_num = []
marital_status= []
occupation = []
relationship = []
race = []
sex = []
capital_gain = []
capital_loss = []
hours_per_week = []
native_country = []
alary = []
#%% load 資料
for n,l in enumerate(reader):
    age.append(l[0])
    workclass.append(l[1])
    fnlwgt.append(l[2])
    education.append(l[3])
    education_num.append(l[4])    
    marital_status.append(l[5])
    occupation.append(l[6])
    relationship.append(l[7])
    race.append(l[8])
    sex.append(l[9])
    capital_gain.append(l[10])
    capital_loss.append(l[11])
    hours_per_week.append(l[12])
    native_country.append(l[13])
    alary.append(l[14])
    
dic ={'age':age,
      'workclass':workclass,
      'fnlwgt':fnlwgt,
      'education':education,
      'education_num':education_num,
      'marital_status':marital_status,
      'occupation':occupation,
      'relationship':relationship,
      'race':race,
      'sex':sex,
      'capital_gain':capital_gain,
      'capital_loss':capital_loss,
      'hours_per_week':hours_per_week,
      'native_country':native_country,
      'alary':alary
      }
data=pd.DataFrame(dic)

#%%  資料文字轉數字

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_le=pd.DataFrame(dic)
data_le['age'] = data['age']
data_le['workclass'] = labelencoder.fit_transform(data['workclass'])
data_le['fnlwgt'] = data['fnlwgt']
data_le['education'] = labelencoder.fit_transform(data['education'])
data_le['education_num'] = data['education_num']
data_le['marital_status'] = labelencoder.fit_transform(data['marital_status'])
data_le['occupation'] = labelencoder.fit_transform(data['occupation'])
data_le['relationship'] = labelencoder.fit_transform(data['relationship'])
data_le['race'] = labelencoder.fit_transform(data['race'])
data_le['sex'] = labelencoder.fit_transform(data['sex'])
data_le['capital_gain'] = data['capital_gain']
data_le['capital_loss'] = data['capital_loss']
data_le['hours_per_week'] = data['hours_per_week']
data_le['native_country'] = labelencoder.fit_transform(data['native_country'])
data_le['alary'] = labelencoder.fit_transform(data['alary'])
#%% 分割訓練資料和正確答案

X = [data_le['age'],
     data_le['workclass'],
     data_le['fnlwgt'],
     data_le['education'],
     data_le['education_num'],
     data_le['marital_status'],
     data_le['occupation'],
     data_le['relationship'],
     data_le['race'],
     data_le['sex'],
     data_le['capital_gain'],
     data_le['capital_loss'],
     data_le['hours_per_week'],
     data_le['native_country']]

Y = [data_le['alary']]

#%%reshape
X = np.array(X) 
Y = np.array(Y)
X = X.reshape(32561,14)
Y = Y.reshape(32561,1)
for i in range(len(X)):
    X[i] = X[i].astype('float32')/255
    
#%%
X_train = []
X_test = []
y_train =[]
y_test = []
#資料分割成測試和訓練
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
#建立分類
clf = DecisionTreeClassifier(criterion='gini',max_depth=4,max_leaf_nodes=5).fit(X_train,y_train)
path = clf.cost_complexity_pruning_path(X_train,y_train)
print(clf.score(X_test,y_test))
#預測
test_y_predicted  = clf.predict(X_test)
for i in range(len(test_y_predicted)):
    print(test_y_predicted[i],y_test[i])
    
    
import pydotplus

dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('iris.pdf')
'''
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()
'''