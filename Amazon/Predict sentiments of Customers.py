# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\Ayush\OneDrive\Desktop\Amazon/Reviews.csv")
df.head()
df.columns
df['Helpful%']=np.where(df['HelpfulnessDenominator']>0,df['HelpfulnessNumerator']/df['HelpfulnessDenominator'],-1)
df['Helpful%'].unique()
df['%upvote']=pd.cut(df['Helpful%'],bins=[-1,0,0.2,0.4,0.6,0.8,1],labels=['Empty','0-20%','20-40%','40-60%','60-80%','80-100%'])
df.head()
# Analysis the upvote of different scores
df.groupby(['Score','%upvote']).agg({'Id':'count'})

df_s=df.groupby(['Score','%upvote']).agg({'Id':'count'}).reset_index()
pivot= df_s.pivot(index='%upvote',columns='Score')

import seaborn as sns
sns.heatmap(pivot,annot=True,cmap='YlGnBu')

# Apply Bag of words NLP

df['Score'].unique()
df2=df[df['Score']!=3]
X=df2['Text']
df2['Score'].unique()
y_dict={1:0,2:0,4:1,5:1}
y=df2['Score'].map(y_dict)

from sklearn.feature_extraction.text import CountVectorizer
c=CountVectorizer(stop_words='english')
X_c =c.fit_transform(X)
X_c.shape

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_c,y)
X_train.shape

from sklearn.linear_model import LogisticRegression
l=LogisticRegression()
ml=l.fit(X_train,y_train)
ml.score(X_train,y_train)
ml.score(X_test,y_test)

w=c.get_feature_names()
coef=ml.coef_.tolist()[0]
coef_df=pd.DataFrame({'Word':w,'Coefficient':coef})
coef_df
coef_df=coef_df.sort_values(['Coefficient','Word'],ascending=False)
coef_df.head()
coef_df.tail()

def text_fit(X,y,nlp_model,ml_model,coef_show=1):
    X_c=nlp_model.fit_transform(X)
    print("features:{}".format(X_c.shape[1]))
    X_train,X_test,y_train,y_test = train_test_split(X_c,y)
    ml=ml_model.fit(X_train,y_train)
    acc= ml.score(X_test,y_test)
    print(acc)
    if coef_show==1:
        w=c.get_feature_names()
        coef=ml.coef_.tolist()[0]
        coef_df=pd.DataFrame({'Word':w,'Coefficient':coef})
        coef_df=coef_df.sort_values(['Coefficient','Word'],ascending=False)
        print('\n')
        print("Top 20 Positive words")
        print(coef_df.head(20))
        print('\n')

text_fit(X, y, c,l)

from sklearn.metrics import confusion_matrix,accuracy_score
def predict(X,y,nlp_model,ml_model):
    X_c=nlp_model.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X_c,y)
    ml=ml_model.fit(X_train,y_train)
    prediction=ml.predict(X_test)
    m=confusion_matrix(prediction,y_test)
    print(m)
    acc= accuracy_score(prediction, y_test)
    print(acc)
    
predict(X, y, c, l)

from sklearn.dummy import DummyClassifier
c=CountVectorizer()
text_fit(X,y,c,DummyClassifier(),0)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english')
text_fit(X,y,tfidf,l,0)
predict(X, y, c, l)


df.head()
data=df[df['Score']==5]
data.head()
data['%upvote'].unique()

data2=data[data['%upvote'].isin(['80-100%', '60-80%', '20-40%', '0-20%'])]
data2.head()
X=data2['Text']
data2['%upvote'].unique()
y_dict={'80-100%':1,'60-80%':1,'20-40%':0,'0-20%':0}
y=data['%upvote'].map(y_dict)
y.value_counts()

tf=TfidfVectorizer()
X_c=tf.fit_transform(X)
X_c.shape
# handling imbalance situations

from imblearn.over_sampling import RandomOverSampler
os= RandomOverSampler()
X_train_res,y_train_res=os.fit_resample(X_c,y)
X_train_res.shape
y_train_res.shape

from collections import Counter
# print("Original:{}".format(Counter(y)))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
np.arange(-2,3)
grid={'C':10.0**np.arange(-2,3),'penalty':['l1','l2']}
log_class=LogisticRegression()
gs= GridSearchCV(estimator=log_class,param_grid=grid,cv=5,n_jobs=-1,scoring='f1_macro')
gs.fit(X_train_res,y_train_res)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_c,y)
gs.predict(X_test)
