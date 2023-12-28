import pandas as pd
import torch
import torch.nn as nn

columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

data = pd.read_csv('heart+disease/processed.cleveland.data',delimiter=',',names=columns)

print (data.head())

# print (data.dtypes)

# print (data[['ca','thal']])

# data[['ca','thal']] = data[['ca','thal']].apply(pd.to_numeric)
# data['ca'] = pd.to_numeric(data.ca,errors='coerce')
# data['thal'] = pd.to_numeric(data.thal,errors='coerce')

cat_features = []

for i in data.columns:
    if (len(data[i].value_counts())<5):
        print (i," : ",data[i].value_counts())
        cat_features.append(i)
# print (cat_features)

data = data.drop(data[data['thal']=='?'].index)

df = pd.get_dummies(data,columns=cat_features,prefix = cat_features)
#
print (df.head())

# sex, cp, fbs, restecg, exang, slope, ca, thal,

print (data.isnull().sum())
print (data['num'])
# print (data.dtypes)