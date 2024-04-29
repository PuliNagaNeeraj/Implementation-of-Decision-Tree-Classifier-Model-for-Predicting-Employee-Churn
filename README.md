# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.Assign the train dataset and test dataset.
6.From sklearn.tree import DecisionTreeClassifier.Use criteria as entropy.
7.From sklearn import metrics.Find the accuracy of our model and predict the require values.
Programs :
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PULI NAGA NEERAJ
RegisterNumber:  212223240130
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
```
## Output:
![image](https://github.com/Abburehan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849336/2f8ae172-9105-42eb-aa10-31a6b4cfa0f5)
```
data.isnull().sum()
```
## Output:
![image](https://github.com/Abburehan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849336/16a2caba-953f-4e6b-b1a2-9c9e35664637)
```
data["left"].value_counts
```
## Output:
![image](https://github.com/Abburehan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849336/78d0b7ed-8253-4bb2-8df8-7ee0fa6fae5f)
```
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
## Output:
![image](https://github.com/Abburehan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849336/d5ad6678-6800-4c28-b9ef-e7be679b6958)
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
## Output:
![image](https://github.com/Abburehan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849336/982f5c63-bdc0-49f7-a081-e50898c77410)
```
y=data["left"]
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
```
```
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
```
## Output:
![image](https://github.com/Abburehan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849336/14ed991e-0579-4e6e-9517-4e274e4b75a8)
```
y_pred = dt.predict(x_test)
from sklearn import metrics
```
```
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/Abburehan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849336/30c2213c-9a66-43c9-8253-2b153a942ebd)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![image](https://github.com/Abburehan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849336/38ee663e-b965-481c-ac56-fdff173315ce)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
