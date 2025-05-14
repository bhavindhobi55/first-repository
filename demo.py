import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\data.csv")
print(data.columns)
print(data.head())
x = data[["country","age","salary"]].values
print(x)
print('---------------------------------------')
y = data[["purchase"]].values
print(y)
print(data.columns)

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)
print('---------------------------------------')

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)
print('---------------------------------------')

label_encode_x = LabelEncoder()
x[:,0]=label_encode_x.fit_transform(x[:,0])
print(x)
print('---------------------------------------')


label_encode_y = LabelEncoder()
y = label_encode_y.fit_transform(y)
print(y)
print('---------------------------------------')

onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(data.country.values.reshape(-1,1)).toarray()
print(x)
print('---------------------------------------')

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
print("train x : ",x_train)
print("test x : ",x_test)
print("train y : ",y_train)
print("test y : ",y_test)