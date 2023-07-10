from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
titanic_train=pd.read_csv('C:\\Users\\abhiy\\Desktop\\BharatIntern Tasks\\Titanic_Classification\\train.csv')
titanic_test=pd.read_csv('C:\\Users\\abhiy\\Desktop\\BharatIntern Tasks\\Titanic_Classification\\test.csv')   
titanic_train.head()    
titanic_train.shape
print(titanic_train['Survived'].value_counts())    
print(titanic_train['Survived'].value_counts().keys())    
plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Survived'].value_counts().keys()),list(titanic_train['Survived'].value_counts()),color=["red","green"])
plt.title("Survived")
plt.show()
print(titanic_train['Pclass'].value_counts())
plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Pclass'].value_counts().keys()),list(titanic_train['Pclass'].value_counts()),color=["orange","blue","black"])
plt.title("Passesnger Class")
plt.show()
print(titanic_train['Sex'].value_counts())
plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Sex'].value_counts().keys()),list(titanic_train['Sex'].value_counts()),color=["red","blue"])
plt.title("Sex")
plt.show()
plt.figure(figsize=(5,7))
plt.hist(titanic_train['Age'],color=["purple"])
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Age Group")
plt.show()
titanic_train["Survived"].isnull() 
print(sum(titanic_train['Survived'].isnull()))
titanic_train['Age'].isnull()  
print(sum(titanic_train['Age'].isnull()))
titanic_train=titanic_train.dropna() 
print(sum(titanic_train['Survived'].isnull())) 
print(sum(titanic_train['Age'].isnull()))  
x_train=titanic_train[["Age"]]
y_train=titanic_train[["Survived"]]
dtc=DecisionTreeClassifier() 
dtc.fit(x_train,y_train)  
print(sum(titanic_test['Age'].isnull()))
titanic_test=titanic_test.dropna()
print(sum(titanic_test['Age'].isnull()))
x_test=titanic_test[['Age']]
y_pred=dtc.predict(x_test)  
print(y_pred) 


