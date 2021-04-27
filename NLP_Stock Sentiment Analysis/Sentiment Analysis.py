# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:25:06 2021

@author: Ayush
"""

import pandas as pd
df=pd.read_csv(r'C:\Users\Ayush\OneDrive\Desktop\NLP_Stock Sentiment Analysis/Data.csv',encoding='ISO-8859-1')
df.head()

train=df[df['Date']<'20150101']
test=df[df['Date']>'20141231']

data=train.iloc[:,2:27]
data.head()
data.columns

data.replace('[^a-zA-Z]',' ',inplace=True)
data.head()

data.columns
new_index=[str(i) for i in range(25)]
new_index
data.columns=new_index
data.head()
data.index
data['0']

for i in new_index:
    data[i]=data[i].str.lower()
data.head()

data.iloc[1,:]

headline=[]
for i in data.iloc[1,:]:
    headline.append(i)
' '.join(headline)

' '.join([str(i) for i in data.iloc[1,:]])

headlines=[]
for r in range(0,len(data)):
    headlines.append(' '.join([str(i) for i in data.iloc[r,:]]))
headlines[0]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

cv=CountVectorizer(ngram_range=(2,2))
traindata_x=cv.fit_transform(headlines)

rc=RandomForestClassifier(n_estimators=200,criterion='entropy')
rc.fit(traindata_x,train['Label'])

test.head()

test_transform=[]

for r in range(0,len(test)):
    test_transform.append(' '.join([str(i) for i in test.iloc[r,2:]]))
    
test_data=cv.transform(test_transform)
pred=rc.predict(test_data)
pred

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(test['Label'],pred)
cm
acc=accuracy_score(test['Label'],pred)
acc

import matplotlib.pyplot as plt
import numpy as np
plt.imshow(cm,cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
labels=['positive','negative']
tick=np.arange(len(labels))
plt.xticks(tick,labels,rotation=45)
plt.yticks(tick,labels)
plt.tight_layout()
plt.xlabel('True Label')
plt.ylabel('Predicted Label')

report= classification_report(test['Label'],pred)
report

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(traindata_x,train['Label'])
pred=nb.predict(test_data)
pred

cm2=confusion_matrix(test['Label'],pred)
cm2
acc=accuracy_score(test['Label'],pred)
acc

import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm,title='Confusion Matrix'):
    
    plt.imshow(cm,cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    labels=['positive','negative']
    tick=np.arange(len(labels))
    plt.xticks(tick,labels,rotation=45)
    plt.yticks(tick,labels)
    plt.tight_layout()
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')

plot_confusion_matrix(cm2)
report= classification_report(test['Label'],pred)
report
