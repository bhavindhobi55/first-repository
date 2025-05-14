#decision tree classifier with gini index 
 
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# import matplotlib.pyplot as plt

# data = pd.read_csv("tennis.csv")
# print(data)

# outlook = LabelEncoder()
# humidity = LabelEncoder()
# windy = LabelEncoder()
# play = LabelEncoder()

# data['outlook'] = outlook.fit_transform(data['outlook'])
# data['humidity'] = outlook.fit_transform(data['humidity'])
# data['windy'] = outlook.fit_transform(data['windy'])
# data['play'] = outlook.fit_transform(data['play'])
# print(data)

# feature_cols = ['outlook','humidity','windy']
# x = data[feature_cols]
# y = data.play
# print(x)
# print(y)

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# classifier = DecisionTreeClassifier(criterion='entropy')   # here write entropy insted gini then it gives entropy decision tree classification
# classifier.fit(x_train,y_train)
# a = classifier.predict(x_test)
# print(a)
# print(x_test)       #0=no 1=yes   outlook[2=sunny,0=overcast,1=rainy]

# print(classifier.score(x_test,y_test))
# print(tree.plot_tree(classifier))
# plt.show()

#----------------------------------------------------------------------

#Random Forest Algorithms
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# iris = datasets.load_iris()
# print(iris)
# print(iris.target_names)
# print(iris.feature_names)
# x = pd.DataFrame({'sepal length':iris.data[:,0],
#               'sepal width':iris.data[:,1],
#               'petal length':iris.data[:,2],
#               'petal width':iris.data[:,3],
#               'species':iris.target})
# print(x.head())
# a = x[['sepal length', 'sepal width', 'petal length', 'petal width']]
# b = x['species']
# print(a)
# print(b)

# x_train,x_test,y_train,y_test = train_test_split(a,b,test_size=0.3,random_state=42)

# clf = RandomForestClassifier(n_estimators = 100,criterion = 'gini')
# clf.fit(x_train,y_train)
# print(clf.predict(x_test))
# print(x_test)
# print("pehlo score with patel width :",clf.score(x_test,y_test))
# print(clf.predict([[3,4,6,8]]))   

# feature_imp = pd.Series(clf.feature_importances_,index = iris.feature_names).sort_values(ascending=False)
# print(feature_imp)
# xb = x[['sepal length', 'sepal width', 'petal length']]
# yb = x['species']

# x_train,x_test,y_train,y_test = train_test_split(xb,yb,train_size=0.3,random_state=42)
# clf = RandomForestClassifier(n_estimators = 100,criterion = 'gini')
# clf.fit(x_train,y_train)
# print(clf.predict(x_test))
# print("bijo score petal width vagar :",clf.score(x_test,y_test))

#----------------------------------------------------------------------

#Random Forest Algorithms
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
 
# dataset = pd.read_csv('Social_Network_Ads.csv')
# print(dataset)

# x = dataset.iloc[:,[1,2,3]].values
# print(x)
# y = dataset.iloc[:,-1].values
# print(y)

# le = LabelEncoder()
# x[:,0] = le.fit_transform(x[:,0])
# print(x)

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# print(len(x_train))
# print(len(x_test))

# model = GaussianNB()
# model.fit(x_train,y_train)
# print(model.predict(x_test))        #purchase =1  ,  not purchase =0
# print(x_test)       #1 = male , 0=female
# print(model.score(x_test,y_test))

#----------------------------------------------------------------------

#Support vector machine Algorithm
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC

# dataset = pd.read_csv("iris.csv")
# print(dataset)

# x = dataset.iloc[:,0:4]
# y = dataset.iloc[:,4]
# print(x)
# print(y)

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# model = SVC(kernel="rbf")
# model.fit(x_train,y_train)
# print(model.predict(x_test))
# print(x_test)

# print(model.score(x_test,y_test))