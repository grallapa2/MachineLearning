import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read dataset

dataset = pd.read_csv("Salary_Data.csv")

# divide dataset into x and y

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# splitting data into training and test datasets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#implement our classifier based on simple linear regression

from sklearn.linear_model import LinearRegression
simplelinearregression = LinearRegression()
simplelinearregression.fit(x_train,y_train)

y_predict = simplelinearregression.predict(x_test)

#y_predict_val = simplelinearregression.predict(11)

#implement the graph
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train,simplelinearregression.predict(x_train))
plt.show()