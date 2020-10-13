# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:35:32 2020

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

#%%
train_dataset = 'abalone.data'
reader = csv.reader(open(train_dataset))

#%%
sex=[]
length = []
Diameter = []
Height = []
Whole_weight = []
Shucked_weight = []
Viscera_weight = []
Shell_weight = []
Rings = []
#%%
for n,l in enumerate(reader):
    sex.append(l[0])
    length.append(l[1])
    Diameter.append(l[2])
    Height.append(l[3])
    Whole_weight.append(l[4])    
    Shucked_weight.append(l[5])
    Viscera_weight.append(l[6])
    Shell_weight.append(l[7])
    Rings.append(float(int(l[8])+1.5))
    
dic ={'sex':sex,
      'length':length,
      'Diameter':Diameter,
      'Height':Height,
      'Whole_weight':Whole_weight,
      'Shucked_weight':Shucked_weight,
      'Viscera_weight':Viscera_weight,
      'Shell_weight':Shell_weight,
      'Rings':Rings
      }
data=pd.DataFrame(dic)
feature_name=[]
for R in Rings:
    if R not in feature_name:
        feature_name.append(str(R))
feature_name.sort()

#%%
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_le=pd.DataFrame(dic)
data_le['sex'] = labelencoder.fit_transform(data['sex'])
data_le['length'] = data['length']
data_le['Diameter'] = data['Diameter']
data_le['Height'] = data['Height']
data_le['Whole_weight'] = data['Whole_weight']
data_le['Shucked_weight'] = data['Shucked_weight']
data_le['Viscera_weight'] = data['Viscera_weight']
data_le['Shell_weight'] = data['Shell_weight']
data_le['Rings'] = labelencoder.fit_transform(data['Rings'])
#%%

X = [data_le['sex'],
    data_le['length'],
    data_le['Diameter'],
    data_le['Height'],
    data_le['Whole_weight'],
    data_le['Shucked_weight'],
    data_le['Viscera_weight'],
    data_le['Shell_weight'],
    ]

Y = [data_le['Rings']]

#%%
X = np.array(X) 
Y = np.array(Y)

X = X.reshape(X.shape[1],X.shape[0])
Y = Y.reshape(Y.shape[1],Y.shape[0])

#%%
X_train = []
X_test = []
y_train =[]
y_test = []

#資料分割成測試和訓練
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
#建立分類
clf = DecisionTreeClassifier(criterion='gini',max_depth=5,max_leaf_nodes=5).fit(X_train,y_train)
#修檢
path = clf.cost_complexity_pruning_path(X_train,y_train)

tree_d = clf.get_depth()
tree_n = clf.get_n_leaves()
#預測

test_y_predicted  = clf.predict(X_test)
for i in range(len(test_y_predicted)):
    with open('abalone.csv','a',newline='') as csvfile:
            writer =csv.writer(csvfile)
            writer.writerow([test_y_predicted[i],y_test[i][0]])
    csvfile.close()

#test_y_predicted  = clf.predict(X_test)
#print(test_y_predicted)
#print(y_test)
#%%劃出決策樹

feature_name = []
class_names =[]
for i in dic.keys():
    if i !='Rings':
        feature_name.append(i)
for i in range(30):
    class_names.append(str(i))
#,out_file='tree.jpg',feature_names= feature_name,class_names = class_names,filled=True,rounded=True
import graphviz
import pydotplus

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_name, class_names=feature_name)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(f'abalone{tree_n}{tree_d}.pdf')
