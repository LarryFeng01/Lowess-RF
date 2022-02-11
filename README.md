# Locally Weighted Regression and Random Forest

## Introduction

In this project I will be presenting the concepts of Locally Weighted Regression and Random Forest Regression. I will explain the theory behhind the regression methods and then apply them to the "Cars.csv" data set where only on input and one output variable is considered. For both methods, the final crossvalidated mean square errors are reported to compare which method achieves better results.

## Theory

### Locally Weighted Regression
Model-based methods use data to buld a parameterized model. And after training the model, it is used for predictions and the data are generally discarded. But, memory-based methods are non-parametric in that they keep the training data and use it any time a prediction is made. Locally weighted regression (lowess) is a memory-based method and performs a regression using the local training data around the point of interest.

Using a kernel, data points are weighted by proximity to the current x location. Then, a regression is computed using the weighted points. 

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
So, the predictions we make are actually a linear combination of the actual observed values of the dependent variable. And for lowess. y hat is obtained as a different linear combination of the values of y.
### Random Forest Regression
Random Forest is a variable of supervised learning algorithm that uses ensemble methods for regression (and also classification). It begins by constructing many decision trees with the training data and outputs the mean prediction of individual trees. Within the random forest, there isn't interaction between the individual trees so the trees prevent each other from individual errors. So, the forest acts as an estimator algorithm that aggregates the result of many trees and then outputs the optimal result.

Since Random Forest is an ensemble technique that can do classificaion and regression, we call the techniques RF uses Bootstrap and Aggregation, which is also known as "bagging". This technique combines multiple decision trees in determining the final output, which was discussed above. 

To start the using Random Forest, we follow the steps to a normal machine learning algorithm:
1) Obtain the data and make sure it is in an accessible format
2) Create a machine learning model
3) Set the baseline model that you want to achieve
4) Train the data to the model
5) Compare the performace of the test data to the predicted data
6) Change parameters around until you are satisfied with the model
7) Interpret the results and report accordingly
8) 
## Modeling Approach
