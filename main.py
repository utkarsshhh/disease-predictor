import pandas as pd
import torch
import torch.nn as nn

columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

data = pd.read_csv('heart+disease/processed.cleveland.data',delimiter=',',names=columns)

print (data.head())

print (data.dtypes)

print (data[['ca','thal']])

# data[['ca','thal']] = data[['ca','thal']].apply(pd.to_numeric)
data['ca'] = pd.to_numeric(data.ca,errors='coerce')
data['thal'] = pd.to_numeric(data.thal,errors='coerce')


print (data.dtypes)