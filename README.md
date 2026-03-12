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
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Siva R
RegisterNumber:  212225100050
*/
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("drive/MyDrive/ML/Salary.csv")
data.head()

<img width="428" height="264" alt="image" src="https://github.com/user-attachments/assets/3ed9dfa1-68f3-4ad2-82f3-ff43fc3afc8a" />
data.info()
<img width="458" height="249" alt="image" src="https://github.com/user-attachments/assets/0983840f-67d1-4b3f-a0fd-b9363bd47631" />
data.isnull().sum()
<img width="247" height="222" alt="image" src="https://github.com/user-attachments/assets/c1a1604c-3f2b-463b-a825-3920df2eeb63" />
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
<img width="358" height="258" alt="image" src="https://github.com/user-attachments/assets/d6ae9e21-835a-4e23-9498-3e8337d82e84" />
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
<img width="241" height="299" alt="image" src="https://github.com/user-attachments/assets/7f7c9c9c-cb6c-45ae-9914-f10a9278996e" />
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
<img width="364" height="49" alt="image" src="https://github.com/user-attachments/assets/75032c33-718e-470f-b025-dc202f0829d7" />
from sklearn import metrics
r2=metrics.r2_score(y_test,y_pred)
r2
<img width="285" height="45" alt="image" src="https://github.com/user-attachments/assets/0ac8ab30-121b-43dd-a416-1ccc1c294aea" />
dt.predict([[5,6]])

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
