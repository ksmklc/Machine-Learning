# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:21:12 2020

@author: Kasım KILIÇ
Tıp Dönem 1
İstanbul Tıp Fakültesi no: 0101190064 

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import plotting
from scipy import stats
from pandas import DataFrame 
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn import tree
import sklearn




asd = pd.read_csv("data.csv")
asd = asd.drop(['Unnamed: 32','id'],axis = 1)


# Buralarda veri inceleme methodlarından bazılarını denemek istedim.
m = plt.hist(asd[asd["diagnosis"] == "M"].radius_mean,bins=30,fc = "r",label = "Malignant")
b = plt.hist(asd[asd["diagnosis"] == "B"].radius_mean,bins=30,fc = "g",label = "Bening")

x = m[0].max()
y = list(m[0]).index(x)
z = m[1][y]
print("En sık görülen kanserli göğüs çapı: "+ str(z)+" birim ?")


"""

Hocam data çıkarma üzerine bir kaç bişey okudum outliers diye bir ifade vardı,
verinin bu genel grubdan çok uzakta olan birimlerini tespit etmek için kullanılmıştı,
verinin değerlerini sıralayıp çeyrek dilimlerindeki ilk elemanlarını veren .describe()
fonksiyonunun kullanarak yapılmıştı bende ona baktım. Ama hiç kullanmadım :D

"""

m1 = asd[asd["diagnosis"] == "M"]
b1 = asd[asd["diagnosis"] == "B"]
desc_b = b1.radius_mean.describe()
q1= desc_b[4]
q3= desc_b[6]
iqr = q3 - q1
alt_sınır = q1 - iqr*1.5
üst_sınır = q3 + iqr*1.5
outlier = b1[(b1.radius_mean<alt_sınır) | (üst_sınır<b1.radius_mean)].radius_mean.values
print("Outliers:" , outlier)
# Describe fonksiyonu ile veri setinin genel özelliklerine baktım.
print(desc_b)



"""
f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(asd.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()
"""


# Her seferinde çalışmasın diye grafikleri tırnak içine aldım
# burada correalasyon değerleri 0.9 dan büyük olanları çıkaracağım
asd = asd.drop(["perimeter_mean","area_mean","radius_worst","area_worst","perimeter_worst","texture_worst","compactness_worst","concavity_worst","concave points_worst","perimeter_se","area_se"],axis=1)

"""

f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(asd.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()

"""
# Burada bazı aşamalardan geçirdiğimiz veri setimizi hazır kütüphanelerden çektiğimiz Makine öğrenmesi fonksiyonlarına sokucaz.
"""
#predict'de notfittederror alıyordum veri tipini string olarak değil de int64 olarak yaparsam , yani tolga hocanın örneği gibi M ve B değil de 1 ve 0 şeklinde,
#düzeleceğini ummuştum ama "x_train,y_train" verilerini fit etmeyi unutmuşum o yüzden bu kısım gereksiz kaldı tırnak içine aldım.
y1=[]
for i in asd["diagnosis"]:
    if (i == "M"):
        i = 1
        y1.append(int(i))
    else:
        i = 0
        y1.append(int(i))
y2= np.array(y1)
y2= y2.astype(np.int64)
"""      
x = asd.iloc[:, 1:].values
y = asd.iloc[:,:1]

func= MinMaxScaler()

x1 = func.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x1,y, test_size=0.2)

func1= tree.DecisionTreeClassifier()

func2= func1.fit(x_train,y_train)
eşleşme= func2.predict(x_test)
print(classification_report(y_test,eşleşme))
