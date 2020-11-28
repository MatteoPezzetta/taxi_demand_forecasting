import pandas as pd
import numpy as np
import math
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve, GridSearchCV

#import matplotlib.pyplot as plt
from matplotlib import pyplot

# Defining data URLs

TRAIN_DATA_URL_1 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_6_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_1 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_6_0703_NWA_Weather_NOFFT_Hourly.csv"

TRAIN_DATA_URL_2 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_5_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_2 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_5_0703_NWA_Weather_NOFFT_Hourly.csv"

TRAIN_DATA_URL_3 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_4_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_3 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_4_0703_NWA_Weather_NOFFT_Hourly.csv"

TRAIN_DATA_URL_4 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_4_2015_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_4 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_4_2015_0703_NWA_Weather_NOFFT_Hourly.csv"

TRAIN_DATA_URL_5 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_5_2015_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_5 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_5_2015_0703_NWA_Weather_NOFFT_Hourly.csv"

TRAIN_DATA_URL_6 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_6_2015_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_6 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_6_2015_0703_NWA_Weather_NOFFT_Hourly.csv"

# Read data from csv files on a pandas dataframe

train_dataset_1 = pd.read_csv(TRAIN_DATA_URL_1, header = 0)
test_dataset_1 = pd.read_csv(TEST_DATA_URL_1, header = 0)

train_dataset_2 = pd.read_csv(TRAIN_DATA_URL_2, header = 0)
test_dataset_2 = pd.read_csv(TEST_DATA_URL_2, header = 0)

train_dataset_3 = pd.read_csv(TRAIN_DATA_URL_3, header = 0)
test_dataset_3 = pd.read_csv(TEST_DATA_URL_3, header = 0)

train_dataset_4 = pd.read_csv(TRAIN_DATA_URL_4, header = 0)
test_dataset_4 = pd.read_csv(TEST_DATA_URL_4, header = 0)

train_dataset_5 = pd.read_csv(TRAIN_DATA_URL_5, header = 0)
test_dataset_5 = pd.read_csv(TEST_DATA_URL_5, header = 0)

train_dataset_6 = pd.read_csv(TRAIN_DATA_URL_6, header = 0)
test_dataset_6 = pd.read_csv(TEST_DATA_URL_6, header = 0)

# Concatenation of dataset for learning and testing
train_dataset = pd.concat([train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4, train_dataset_5, train_dataset_6]) #provare a togliere June 2016 (train dataset 1)
test_dataset = pd.concat([test_dataset_6])
print(test_dataset.shape)

# Extracting the label feature
train_labels = train_dataset.pop('Label_Data')
test_labels = test_dataset.pop('Label_Data')

print(test_labels[:100])

print("length of test labels is: {}".format(len(test_labels)))

# Define the sMAPE (Symmeric Mean Absolute Percentage Error)
def s_mean_absolute_percentage_error(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred + 1)))*100

# Build the XGBoost regression model
def xgboost_reg(train_data, train_true, test_data, test_true):

	# hyper-parameter tuning
	hyper_parameter = {"max_depth":[7], "n_estimators":[10, 20, 30, 40, 50, 60, 70, 80, 90]} # depth of the trees and number of trees
	clf = xgb.XGBRegressor() # XGBRegressor -> estimator ? Here I don't know yet the hyperparameters. I first need to find the optimal with GridSearchCV
	best_parameter = GridSearchCV(clf, hyper_parameter, scoring = "neg_mean_absolute_error", cv = 3) # construct the GridSearchCV object best_parameter
	best_parameter.fit(train_data, train_true) # fit gradient boosting regressor. 'fit' is training the model hyperparameters so that to find the optimal hyperparameters
	estimators = best_parameter.best_params_["n_estimators"]
	depth = best_parameter.best_params_["max_depth"]

	print("The number of estimators is = {}".format(estimators))
	print("The depth of the model is = {}".format(depth))

	# appltying xgboost regressor with best hyper-parameter
	clf = xgb.XGBRegressor(max_depth = depth, n_estimators = estimators, objective = 'reg:linear') # here I am applying what I found to the gradient boosting model
	
	eval_metric = ['error', 'logloss'];
	eval_set = [(train_data, train_true), (test_data, test_true)]

	history = clf.fit(train_data, train_true, eval_metric = eval_metric, eval_set = eval_set, verbose = True) # we fit clf to the training set
	
	train_pred = clf.predict(train_data)
	results = clf.evals_result()
	epochs = len(results['validation_0']['error'])
	x_axis = range(0, epochs)

	#train_MAPE = mean_absolute_error(train_true, train_pred)/(sum(train_true)/len(train_true)) # literature says that MAPE is not a good indicator, check why
	train_sMAPE = s_mean_absolute_percentage_error(train_true, train_pred)
	train_MSE = mean_squared_error(train_true, train_pred)
	train_MAE = mean_absolute_error(train_true, train_pred)
	
	test_pred = clf.predict(test_data) # this gives us the prediction using the test data
	#test_MAPE = mean_absolute_error(test_true, test_pred)/(sum(test_true)/len(test_true))
	test_sMAPE = s_mean_absolute_percentage_error(test_true, test_pred)
	test_MSE = mean_squared_error(test_true, test_pred)
	test_MAE = mean_absolute_error(test_true, test_pred)


	# Ragional sMAPE (per each Cluster)
	smape = s_mean_absolute_percentage_error(test_true[:450], test_pred[:450])
	print('Test sMAPE Cluster1: %.3f' % smape)

	for i in range(0,29): # starts from 0 but I am actually computing starting from cluster n.1
		smape = s_mean_absolute_percentage_error(test_true[(450+i*500):((450+i*500)+500)], test_pred[(450+i*500):((450+i*500)+500)])
		print('Test sMAPE Cluster{}: {}'.format((i+2), smape))

	# Regional RMSE (per each Cluster)
	rmse = math.sqrt(mean_squared_error(test_true[:450], test_pred[:450]))
	print('Test RMSE Cluster1: %.3f' % rmse)

	for i in range(0,29): # starts from 0 but I am actually computing starting from cluster n.1
		rmse = math.sqrt(mean_squared_error(test_true[(450+i*500):((450+i*500)+500)], test_pred[(450+i*500):((450+i*500)+500)]))
		print('Test RMSE Cluster{}: {}'.format((i+2), rmse))

	# PLOT RESULTS

	x_axis = np.zeros(100)
	x_axis = np.linspace(24, 124, 100)

	pyplot.figure(0)
	#x_ax = range(len(test_true[:300]))
	pyplot.plot(x_axis, test_true[524:624], color = 'b', marker = 'o', markersize = '4', label="Real data")
	pyplot.plot(x_axis, test_pred[524:624], color = 'g', linestyle = 'dashed', marker = 'o', markersize = '4', label="Predicted (XGBoost)")
	pyplot.grid()
	pyplot.xlabel('Time bin (hours)')
	pyplot.ylabel('Number of pickups')
	pyplot.title("Cluster 2 - Pickups per Hour: Real Data vs Prediction with XGBoost")
	pyplot.legend()
	pyplot.savefig('Result_of_XGBoost_Prediction_vs_TrueData_first_100_HQ', dpi = 300)

	# Return prediction performance
	return train_MAE, train_sMAPE, train_MSE, test_MAE, test_sMAPE, test_MSE

# next I call the XGBoost with only test_3 data for testing
trainMAE_xgb, train_sMAPE_xgb, trainMSE_xgb, testMAE_xgb, test_sMAPE_xgb, testMSE_xgb = xgboost_reg(train_dataset, train_labels, test_dataset, test_labels)

# Print prediction performance on both train and test datasets
print("trainMAE_xgb = {}". format(trainMAE_xgb))
print("train_sMAPE_xgb = {}".format(train_sMAPE_xgb))
print("trainMSE_xgb = {}".format(trainMSE_xgb))
print("testMAE_xgb = {}".format(testMAE_xgb))
print("test_sMAPE_xgb = {}".format(test_sMAPE_xgb))
print("testMSE_xgb = {}".format(testMSE_xgb))