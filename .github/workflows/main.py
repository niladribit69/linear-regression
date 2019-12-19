# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1].values

# Splitting into train and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3)

# Applying simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

sample_output = regressor.predict([[3.5]])
y_pred = regressor.predict(X_test)




#Visualizing the training data
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Sal vs Years')
plt.xlabel('Salary')
plt.ylabel('Years of Experience')
plt.show()

#Visualizing the test data
plt.scatter(X_test,y_test, color = 'red')
plt.scatter(X_test,y_pred, color = 'green')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Sal vs Years')
plt.xlabel('Salary')
plt.ylabel('Years of Experience')
plt.show()

