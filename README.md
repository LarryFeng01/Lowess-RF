# Locally Weighted Regression and Random Forest

## Introduction

In this project I will be presenting the concepts of Locally Weighted Regression and Random Forest Regression. I will explain the theory behhind the regression methods and then apply them to the "Cars.csv" data set where only on input and one output variable is considered. For both methods, the final crossvalidated mean square errors are reported to compare which method achieves better results.

## Theory

### Locally Weighted Regression
Model-based methods use data to buld a parameterized model. And after training the model, it is used for predictions and the data are generally discarded. But, memory-based methods are non-parametric in that they keep the training data and use it any time a prediction is made. Locally weighted regression (lowess) is a memory-based method and performs a regression using the local training data around the point of interest.

Using a kernel, data points are weighted by proximity to the current x location. Then, a regression is computed using the weighted points. 

In class, we compare the mathematical concept of lowess compared to linear regression. The main idea of linear regression is
![render](https://user-images.githubusercontent.com/58920498/153341904-aeef19e6-153d-4158-bb1d-c322d315beae.png)


### Random Forest Regression

## Modeling Approach
