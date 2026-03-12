# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
 6.  import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
``

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

```

## Output:
<img width="390" height="265" alt="image" src="https://github.com/user-attachments/assets/800bcb06-6fcb-4060-ba96-f7ad65acefb1" />
<img width="603" height="237" alt="image" src="https://github.com/user-attachments/assets/a607a7b9-c1a1-4a35-a5cc-b8665e4fa1a9" />
<img width="201" height="88" alt="image" src="https://github.com/user-attachments/assets/f2247213-b015-4922-b264-d092249e6955" />
<img width="323" height="234" alt="image" src="https://github.com/user-attachments/assets/465e8fe9-aade-43c4-bc74-9eaa5a1a4b0d" />
<img width="239" height="38" alt="image" src="https://github.com/user-attachments/assets/d38e9138-bef9-4074-bd6d-7816bf23baa6" />

<img width="1065" height="41" alt="image" src="https://github.com/user-attachments/assets/ca0084d3-6abe-4d0e-906b-f4366b7d16e8" />
<img width="311" height="38" alt="image" src="https://github.com/user-attachments/assets/8ffcfbb2-babd-48f5-a138-493564c37e09" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
