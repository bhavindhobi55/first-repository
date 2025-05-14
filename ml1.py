#split data into train and test with ratio 80-20

# from sklearn.model_selection import train_test_split
# import pandas as pd
# df = pd.read_csv("insurance.csv")
# print(df)
# x = df['age']
# y = df['premium']
# print(x)
# print(y)
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# print("train data of x\n",len(x_train))
# print("test data of x\n",x_test)
# print("train data of y\n",y_train)
# print("test data of y\n",len(y_test))

#---------------------------------------------------------------

#linear regression with single variable

# import seaborn as sns
# import pandas as pd
# from sklearn import linear_model
# import matplotlib.pyplot as plt

# df = pd.read_csv("insurance.csv")
# print(df)
# cv = sns.lmplot(x = 'age',y = 'premium',data = df)
# plt.show()
# print(cv)

# reg = linear_model.LinearRegression()   #prediction of premium according to age
# reg.fit(df[['age']],df['premium'])  
# x = reg.predict([[50]])
# print(x)

# x = reg.coef_   #co-efficient
# print(x)

# x = reg.intercept_  #intercept
# print(x)

#---------------------------------------------------------------

#linear regression with Multiple variable

# import pandas as pd
# from sklearn import linear_model
# import matplotlib.pyplot as plt

# df = pd.read_csv("insurance.csv")
# print(df)

# mean_height = df.height.mean()
# print(df.height)
# df.height = df.height.fillna(mean_height)
# print(df)

# reg = linear_model.LinearRegression()
# reg.fit(df[['age','height','weight']],df['premium'])
# x = reg.coef_
# print(x)
# x = reg.intercept_
# print(x)
# x = reg.predict([[27,167.56,60]])
# print(x)

    #predict the same with formula
    #2150.26052416*27+-248.45851574*167.56+312.65291961*60+-16827.013154824883

#---------------------------------------------------------------

#polynomial regression

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import linear_model
# from sklearn.preprocessing import PolynomialFeatures

# df = pd.read_csv("emp.csv")
# print(df)

# x = df.iloc[:,1:2].values   #allocate values to x and y and show scatter plot
# print(x)
# y = df.iloc[:,2].values
# print(y)    
# plt.scatter(x,y)
# plt.show()

# c = sns.lmplot(x = 'level',y = 'salary',data = df)  #normal regression doesn't give proper output
# print(c)
# plt.show()
# reg = linear_model.LinearRegression()
# reg.fit(x,y)
# a = reg.predict([[6.5]])
# print("normal regression :",a)

# poly = PolynomialFeatures(degree=2)         #polynomial regression give perfect answer
# x_poly = poly.fit_transform(x)
# reg2 = linear_model.LinearRegression()
# reg2.fit(x_poly,y)
# print("polynomial regression :",reg2.predict(poly.fit_transform([[6.5]])))

#---------------------------------------------------------------

#Logistic regression Binary Classification (only two value yes/no)

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# dataset = pd.read_csv("insurance2.csv")
# # print(dataset)
# dataset['insurance'].replace({'no':'0','yes':'1'},inplace = True)
# print(dataset)
# plt.scatter(x='age',y='insurance',data=dataset)
# plt.show()
# train_x,test_x,train_y,test_y = train_test_split(dataset[['age']],dataset['insurance'],test_size=0.2)
# print("train x :",len(train_x))
# print("test x :",len(test_x))
# print("train y :",len(train_y))
# print("test y :",len(test_y))

# lr = LogisticRegression()
# lr.fit(train_x,train_y)
# print(lr.predict(test_x))
# print(lr.predict([[60]]))

#---------------------------------------------------------------

#Logistic regression Multiclass Classification (more than one value)

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import seaborn as sns

# dataset = pd.read_csv("iris.csv")
# print(dataset.head())
# # print(dataset['species'].unique())
# dataset['species'].replace({'Iris-setosa':'1','Iris-versicolor':'2','Iris-virginica':'3'},inplace=True)
# # print(dataset)

# train_x, test_x, train_y, test_y = train_test_split(dataset[['sepal_length','sepal_width','petal_length','petal_width']], dataset['species'], test_size=0.2)

# # print(f"train_x: {len(train_x)}")
# # print(f"train_y: {len(train_y)}")
# # print(f"test_x: {len(test_x)}")
# # print(f"test_y: {len(test_y)}")

# lr = LogisticRegression()           #to predict 
# lr.fit(train_x, train_y)               
# print(lr.predict(test_x))           
# print(test_x)                       

# print(lr.score(test_x,test_y))      #calculate the score
# sns.pairplot(dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']],hue='species')
# plt.show()