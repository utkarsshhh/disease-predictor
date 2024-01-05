import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.optim import SGD
columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

data = pd.read_csv('heart+disease/processed.cleveland.data',delimiter=',',names=columns)

print (data.head())

# print (data.dtypes)

# print (data[['ca','thal']])

data['ca'] = pd.to_numeric(data.ca,errors='coerce')
data['thal'] = pd.to_numeric(data.thal,errors='coerce')

cat_features = []

for i in data.columns:
    if (len(data[i].value_counts())<5):
        # print (i," : ",data[i].value_counts())
        cat_features.append(i)
# print (cat_features)

data = data.drop(data[data['thal']=='?'].index)

df = pd.get_dummies(data,columns=cat_features,prefix = cat_features)
#
# print (df['num'])

# sex, cp, fbs, restecg, exang, slope, ca, thal,

print (df.isnull().sum())
print (df['num'].value_counts())
print (df.dtypes)

print (df.shape)

df.dropna(axis=0,inplace=True)


features = df.drop(labels='num',axis =1)
target = df[['num']]

encoder = OneHotEncoder(sparse=False)

encoded_target=encoder.fit_transform(target)
encoded_target = torch.tensor(encoded_target,dtype=torch.float32)

features = torch.tensor(features.values,dtype=torch.float32)

class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,output_size)
        )

    def forward(self,input):
        return self.model(input)

