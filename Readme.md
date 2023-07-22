# Taiwan Economic Analysis with MATLAB
- Split the data into training sample and testing sample. i.e., use the first 70% of the data as training sample and the remainder as testing sample.
- Do a preliminary covariance analysis on all variables. Plot the heatmap to show the correlation structure of all variables. Clean the data if necessary. Briefly comment on your findings.
- Use the training sample and a simple logistic regression model, including all predictors, to train the model. Use the testing sample to predict company bankruptcy (if estimated probability of bankruptcy is greater or equal to 0.5, then we predict this company will be bankrupt) and show the confusion matrix. Report the accuracy rate for the out-of-sample (OOS) prediction.
- Use the training sample and a logistic regression model which only included the 5 most correlated predictors with the y variable, to train the model. Then, similarly, report the OOS confusion matrix and the accuracy rate.
- Use a boosted classification tree to train the model and then, similarly, report the OOS confusion matrix and the accuracy rate.
- Use a random forest to train the model and then, similarly, report the OOS confusion matrix and the accuracy rate.
- Compare the results from different models. Comment on your findings.