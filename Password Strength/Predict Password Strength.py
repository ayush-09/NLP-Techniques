# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:10:10 2021

@author: Ayush
"""

import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv(r'C:\Users\Ayush\OneDrive\Desktop\Password Strength/data.csv',error_bad_lines=False)
data.head()
data['strength'].unique()
data.isna().sum()
data[data['password'].isnull()]
data.dropna(inplace=True)
data.isnull().sum()
sns.countplot(data['strength'])

password_tuple=np.array(data)
password_tuple

import random
random.shuffle(password_tuple)

x=[l[0] for l in password_tuple]
y=[l[1] for l in password_tuple]
x
y

def word_divide_char(inputs):
    ch=[]
    for i in inputs:
        ch.append(i)
    return ch

from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer(tokenizer=word_divide_char)
X=vec.fit_transform(x)
X.shape
vec.get_feature_names()
first_vec=X[0]
first_vec.T.todense()

df=pd.DataFrame(first_vec.T.todense(),index=vec.get_feature_names(),columns=['TF-IDF'])
df.sort_values(by=['TF-IDF'],ascending=False)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression(random_state=0,multi_class='multinomial')
lg.fit(X_train,y_train)

dt=np.array(['%@123abcd'])
prd=vec.transform(dt)
lg.predict(prd)

y_pred=lg.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)
cm
accuracy_score(y_test,y_pred)
report=classification_report(y_test, y_pred)
report