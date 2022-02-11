# Locally Weighted Regression and Random Forest Project

## Introduction

In this project I will be presenting the concepts of Locally Weighted Regression and Random Forest Regression. I will explain the theory behind the regression methods and then apply them to the "Cars.csv" data set where only on input and one output variable is considered. For both methods, the final crossvalidated mean square errors are reported to compare which method achieves better results.

## Theory

### Locally Weighted Regression
Model-based methods use data to build a parameterized model. And after training the model, it is used for predictions and the data are generally discarded. But, memory-based methods are non-parametric in that they keep the training data and use it any time a prediction is made. Locally weighted regression (lowess) is a memory-based method and performs a regression using the local training data around the point of interest.

Using a kernel, data points are weighted by proximity to the current *x* location. Then, a regression is computed using the weighted points. 

In class, we compare the mathematical concept of lowess compared to linear regression. The main idea of linear regression is
![render](https://user-images.githubusercontent.com/58920498/153341904-aeef19e6-153d-4158-bb1d-c322d315beae.png)

If we multiply all sides of the equation with a matrix of weights in which the matrix is a diagonal matrix): 
![render](https://user-images.githubusercontent.com/58920498/153642302-0fec3155-9737-4af8-b34d-64bd92f507bb.png)

The rows of the matrix *X* are the independent observation, and each row has a number of features we denote as *p*. And, the distance between two independent observations is just the Euclidean distance between the two *p*-dimensional vectors. The equation for the Euclidean distance is: 
![render](https://user-images.githubusercontent.com/58920498/153643283-9746d06f-7722-4799-a8a8-2bda24ee62d9.png)

Now, let's take a step back and analyze how linear regression predicts *y*. 
We start with: 
![render](https://user-images.githubusercontent.com/58920498/153643499-3d9efc09-7ef9-4137-90e7-ec9c79757274.png)

We solve for β: 
![render](https://user-images.githubusercontent.com/58920498/153643757-955ed0b7-c0b9-42f2-aa80-27434e1f7cd8.png)

When we take the expected value of the above equation, epsilon will be 0, so the second half of the equation disappears. 
Now we solve for y:
![render](https://user-images.githubusercontent.com/58920498/153644012-19166fff-db4c-470c-8c65-a0666916947e.png)

These are the predictions that we make with linear regression. But with locally weighted regression, our equation is:
![render](https://user-images.githubusercontent.com/58920498/153644163-484fddb6-1494-4eb8-af4c-0fc21a4ea438.png)

So, the predictions we make from our univariate data are a linear combination of the actual observed values of the dependent variable. And for lowess, y-hat is obtained as a different linear combination of the values of y.

### Random Forest Regression
Random Forest (RF) is a variable of supervised learning algorithm that uses ensemble methods for regression (and also classification). It begins by constructing many decision trees with the training data and outputs the mean prediction of individual trees. Within the random forest, there isn't interaction between the individual trees so the trees prevent each other from individual errors. So, the forest acts as an estimator algorithm that aggregates the result of many trees and then outputs the optimal result.

Since Random Forest is an ensemble technique that can do classificaion and regression, we call the techniques RF uses Bootstrap and Aggregation, which is also known as "bagging". This technique combines multiple decision trees in determining the final output, which was discussed above. 

To start the using Random Forest, we follow the steps to a normal machine learning algorithm:
1) Obtain the data and make sure it is in an accessible format
2) Create a machine learning model
3) Set the baseline model that you want to achieve
4) Train the data to the model
5) Compare the performance of the test data to the predicted data
6) Change parameters around until you are satisfied with the model
7) Interpret the results and report accordingly

## Modeling Approach

Let's import some data. We will be using the '*Cars.csv*' dataset; to import run the `pd.read_csv()` command. 
Running our output will give us a table of our data:
--- | MPG | CYL | ENG | WGT
--- | --- | --- | --- |--- 
0 | 18.0 | 8 | 307.0 | 3504
1 | 15.0 | 8 | 350.0 | 3693
2 | 18.0 | 8 | 318.0 | 3436
... | ... | ... | ... | ...
389 | 32.0 | 4 | 135.0 | 2295
390 | 28.0 | 4 | 120.0 | 2625
391 | 31.0 | 4 | 119.0 | 2720

Since we are modeling for univariate observations, our *x* will be the weight of the car ('*WGT*'), and our *y* will be the miles per gallon ('*MPG*'). 

### Locally Weighted Regression
```
  def lowess_reg(x, y, xnew, kern, tau):
    n = len(x)
    yest = np.zeros(n)  
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)
```
The above function is doing exactly what was described in the theory section where we calculate the weights given the data. Now, we will split the data into train/test splits and standardize the data. We must also be careful about the shape of the data while standardizing and transforming. 
```
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.25, random_state=123)
scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain.reshape(-1,1))
xtest_scaled = scale.transform(xtest.reshape(-1,1))
```
Lastly, we will use the function to get predictions for *y* and compare them with our data.
```
yhat_test = lowess_reg(xtrain_scaled.ravel(),ytrain,xtest_scaled,tricubic,0.1)
mse(yhat_test,ytest)
```
We can also visualize the algorithm by making a scatterplot of the data and adding the predicted lines on top. 
```
Q = np.column_stack([xtest_scaled,yhat_test])
Q = Q[np.argsort(Q[:,0])]
plt.scatter(xtest_scaled,ytest,color='blue', alpha=0.5)
plt.plot(Q[:,0],Q[:,1], color='red', lw=2)
```
![render](https://user-images.githubusercontent.com/58920498/153676341-36aff249-8e62-4be9-ba99-b17cb3a86de1.png)

### Random Forest
Random Forest requires fewer lines to run:
```
rf = RandomForestRegressor(n_estimators=100,max_depth=3)
rf.fit(xtrain_scaled,ytrain)
mse(ytest,rf.predict(xtest_scaled))
```
Let's make a plot for this too.
```
M = np.column_stack([xtest_scaled,yhat_rf])
M = M[np.argsort(M[:,0])]
plt.scatter(xtest_scaled,ytest,color='blue', alpha=0.5)
plt.plot(M[:,0],M[:,1], color='red', lw=2)
```
![render](https://user-images.githubusercontent.com/58920498/153676813-51b3f395-4ce1-4533-902b-90d9fd034b01.png)

However, we can also write all of the above code for both algorithms into one or two blocks to make it concise, and this time we will add in cross validation. 
```
kf = KFold(n_splits=10,shuffle=True,random_state=410)
mse_lwr = []
mse_rf = []
rf = RandomForestRegressor(n_estimators=100,max_depth=3, random_state=410)
kf = KFold(n_splits=10,shuffle=True,random_state=410)
for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_lwr = lowess_reg(xtrain.ravel(),ytrain,xtest.ravel(),tricubic,0.4)
  rf.fit(xtrain,ytrain)
  yhat_rf = rf.predict(xtest)
  mse_lwr.append(mse(ytest,yhat_lwr))
  mse_rf.append(mse(ytest,yhat_rf))
print('The MSE for RF is :' + str(np.mean(mse_rf)))
print('The MSE for locally weighted reg is :' + str(np.mean(mse_lwr)))
```
The last two lines will print the result of the mean squared error, which is what we base our results on. The lower error is better, so in our case Lowess is the better model with a cross validated MSE of 17.765 compared to 17.922 of RF.

## Conclusion
Although there are times when changing the hyperparameters for RF will lead to a lower MSE, changing a hyperparamter in lowess in response can lead to an even lower MSE. Over a couple experiments, lowess consistently had the lower MSE when changing aroud hyperparameters. In conclusion, for the "Cars.csv" data set, my results indicate that lowess returns a lower cross validated mean squared error compared to random forest; therefore, lowess is a better algorithm for our purposes.
