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
#test_dataset = 'adult.test'

reader = csv.reader(open(train_dataset))
#test_reader = csv.reader(open(test_dataset))

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
    with open('adult.csv','a',newline='') as csvfile:
            writer =csv.writer(csvfile)
            writer.writerow([test_y_predicted[i],y_test[i][0]])
    csvfile.close()

import pydotplus
tree_n = clf.get_n_leaves()
tree_d = clf.get_depth()
feature_name=['>=50K','<50K']
dot_data = tree.export_graphviz(clf, out_file=None,class_names=['>=50K','<50K'])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(f'adult{tree_n}{tree_d}.pdf')

