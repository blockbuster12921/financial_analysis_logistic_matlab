%% Task1
% Split the data into training sample and testing sample. i.e., use the first
% 70% of the data as training sample and the remainder as testing sample.

input_file = 'CBP_data 2023.csv';

% Read the CSV file using readmatrix
input_table = readtable(input_file);

% Handle missing data
input_table = rmmissing(input_table);

% Set the training ratio (70%) and testing ratio (30%)
training_ratio = 0.7;
testing_ratio = 0.3;

% Shuffle the rows randomly
rng('default'); % For reproducibility, remove this line if you want different results each time
shuffled_table = input_table(randperm(size(input_table, 1)), :);

% Determine the number of rows for training and testing
num_rows = size(shuffled_table, 1);
num_train = round(training_ratio * num_rows);
num_test = num_rows - num_train;

% Extract the training and test data
train_table = shuffled_table(1:num_train, :);
test_table = shuffled_table(num_train+1:end, :);

% Show the number of rows and columns in the input table, train table, test
% table
disp('Task1 Result:')
disp(size(input_table));
disp(size(train_table));
disp(size(test_table));

%% Task2
% Do a preliminary covariance analysis on all variables. Plot the heatmap to show the
% correlation structure of all variables. Clean the data if necessary. Briefly comment on
% your findings

cor = corrcoef(table2array(train_table));
heatmap(cor);
disp(max(cor(1, :)));
disp(min(cor(1, :)));

%% Task3
% Use the training sample and a simple logistic regression model, including all
% predictors, to train the model. Use the testing sample to predict company
% bankruptcy (if estimated probability of bankruptcy is greater or equal to 0.5,
% then we predict this company will be bankrupt) and show the confusion matrix.
% Report the accuracy rate for the out-of-sample (OOS) prediction.

%% Task4
% Use the training sample and a logistic regression model which only included
% the 5 most correlated predictors with the y variable, to train the model. Then,
% similarly, report the OOS confusion matrix and the accuracy rate.

%% Task5
% Use a boosted classification tree to train the model and then, similarly, report
% the OOS confusion matrix and the accuracy rate.

%% Task6
% Use a random forest to train the model and then, similarly, report the OOS
% confusion matrix and the accuracy rate.

%% Task7
% Compare the results from different models. Comment on your findings.

