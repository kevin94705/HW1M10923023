# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:15:00 2020

@author: USER
"""
import csv
import os
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pandas import DataFrame
#讀資料
train_dataset = 'abalone.data'
reader = csv.reader(open(train_dataset))

sex=[]
length = []
Diameter = []
Height = []
Whole_weight = []
Shucked_weight = []
Viscera_weight = []
Shell_weight = []
Rings = []

for n,l in enumerate(reader):
    sex.append(l[0])
    length.append(l[1])
    Diameter.append(l[2])
    Height.append(l[3])
    Whole_weight.append(l[4])    
    Shucked_weight.append(l[5])
    Viscera_weight.append(l[6])
    Shell_weight.append(l[7])
    Rings.append(l[8])
    #Rings.append(float(int(l[8])+1.5))

dic = {'sex':sex,
      'length':length,
      'Diameter':Diameter,
      'Height':Height,
      'Whole_weight':Whole_weight,
      'Shucked_weight':Shucked_weight,
      'Viscera_weight':Viscera_weight,
      'Shell_weight':Shell_weight,
      'Rings':Rings}
data=pd.DataFrame(dic)
#轉數值 
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
data_le['Rings'] = data['Rings']

arr = np.array(data_le,dtype = float)
np.random.shuffle(arr)
X = arr[:,:8]
Y = arr[:,8]

#X_train =arr[:round(len(arr)/10*8),:]
X_train =[]
X_test = []
Y_train =[]
Y_test = []
#分割訓練資料和測試資料
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
#建立分類器
clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=(3),max_leaf_nodes=(5))
#修剪
clf = clf.fit(X_train,Y_train)
#預測
Y_test_predict = clf.predict(X_test)
#績效計算並輸出
score = clf.score(X_test,Y_test)
print(score)
#accuracy = accuracy_score(Y_test, Y_test_predict)
#print(accuracy)
#%%畫決策樹        
feature_name = ['sex','length','Diameter','Height','Whole_weight','Shucked_weight',
                'Viscera_weight','Shell_weight']
class_name = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',
               '16','17','18','19','20','21','22','23','24','25','26','27','28','29']
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, feature_names= feature_name, 
                                class_names=class_name, filled=True, rounded=True) 
graph = graphviz.Source(dot_data) 
graph.render("iris")
#import pydotplus
#graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf("1.pdf")
#%%將測試資料結果輸出到Excel
df1 = DataFrame(data_le)
df2 = DataFrame(Y, columns=['predict result'])
#df2.to_excel("result2.xlsx", startrow = 0, startcol = 10)
with pd.ExcelWriter('result.xlsx') as writer:
    df1.to_excel(writer)
    df2.to_excel(writer, index=0, startrow=0, startcol=10)
