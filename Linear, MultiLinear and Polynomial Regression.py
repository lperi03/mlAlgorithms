#Linear and Polynomial Regression
#some algorithms will be demoed using scikit-learn, and some will be demoed using the tensorFlow/Keras framework. 
#For regression, building a model in scikit will suffice more often than not

#importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt


#lets say we have a sample dataset, stored in some local path on our device as a csv(comma separated values) file
#reading a sample path into our workspace using pandas and storing it as a dataframe
path = 'C:\Users\Lakshman Peri\OneDrive\Desktop\sampleFile'
df = pd.read_csv(path)


#Linear regression can be defined as y = wx+ b, where w is the weight(slope) and bias(y-intercept). 
#When there are multiple features/variables affecting y(our dependent variable), then w is defined as a weights matrix. This is known as multi-linear regression

#linear regression sample
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#In our dataframe, we will have some features that we want to correlate using regression
#We can define our X(independent) and Y(dependent) variables using columns of data from our dataframe
#rename the columns based on your dataset!

X = df[['independent']]
Y = df[['dependent']]

#fitting the model to our data
lr.fit(X, Y)
#We can then make a prediction using our model
Yhat = lr.predict(X)
#We can find the slope and intercept of our model using the following commands
slope = lr.coef_
intercept = lr.intercept_


#We can also use a plot to visualize the fit made by the model vs the actual data
#x is the independent(predictor) variable and y is our dependent variable.
#replace independent and dependent with your own variables, as defined above!!
plt.figure(figsize = (12, 10))
sns.regplot(x = 'independent', y = 'dependent')
plt.show()

#Model evaluation
#using r^2 to see correlation
lr.score(X, Y)





 

