# Naive Prediction approach where the current value of the Label is used as prediciton for the next time instant

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load the test data
TEST_DATA_URL_6 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_6_2015_0703_NWA_Weather_NOFFT_Hourly.csv"
test_dataset_6 = pd.read_csv(TEST_DATA_URL_6, header = 0)
test_dataset = pd.concat([test_dataset_6])

# Definition of the sMAPE
def s_mean_absolute_percentage_error(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred + 1)))*100

# Naive forecast for 1-step ahead prediction
n_steps = 1
naive_forecast_1step = np.array(test_dataset['Label_Data'][:-n_steps]) #taking the Labels as the predictions
labels_1step = np.array(test_dataset['Label_Data'][n_steps:])

# Compute sMAPE
smape_1step = s_mean_absolute_percentage_error(labels_1step, naive_forecast_1step)
print("sMAPE_1step = {}".format(smape_1step))

# Regional sMAPE (per each Cluster)
smape = s_mean_absolute_percentage_error(labels_1step[:450], naive_forecast_1step[:450])
print('Test sMAPE Cluster1: %.3f' % smape)

for i in range(0,29): # starts from 0 but I am actually computing starting from cluster n.1
	smape = s_mean_absolute_percentage_error(labels_1step[(450+i*500):((450+i*500)+500)], naive_forecast_1step[(450+i*500):((450+i*500)+500)])
	print('Test sMAPE Cluster{}: {}'.format((i+2), smape))

# Regional RMSE (per each Cluster)
rmse = math.sqrt(mean_squared_error(labels_1step[:450], naive_forecast_1step[:450]))
print('Test RMSE Cluster1: %.3f' % rmse)

for i in range(0,29): # starts from 0 but I am actually computing starting from cluster n.1
	rmse = math.sqrt(mean_squared_error(labels_1step[(450+i*500):((450+i*500)+500)], naive_forecast_1step[(450+i*500):((450+i*500)+500)]))
	print('Test RMSE Cluster{}: {}'.format((i+2), rmse))


# Plot to visualize prediction performance
pyplot.figure(0)
pyplot.plot(labels_1step[:100], color = 'b', label = 'Real data')
pyplot.plot(naive_forecast_1step[:100], color = 'c' , linestyle = 'dashed', label = 'Predicted (Naive)')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("New York Pickups per Hour: Real Data vs Prediction with XGBoost")
pyplot.legend()
pyplot.savefig("Naive_prediction_plot_comparison")

'''
DATA_PLOT_1 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_4_0703_NWA_Weather_NOFFT_Hourly.csv"
DATA_PLOT_2 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_5_0703_NWA_Weather_NOFFT_Hourly.csv"
DATA_PLOT_3 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_6_0703_NWA_Weather_NOFFT_Hourly.csv"

dataset_1 = pd.read_csv(DATA_PLOT_1, header = 0)
dataset_2 = pd.read_csv(DATA_PLOT_2, header = 0)
dataset_3 = pd.read_csv(DATA_PLOT_3, header = 0)

Label_Data_1 = np.array(dataset_1['Label_Data'])
Label_Data_2 = np.array(dataset_2['Label_Data'])
Label_Data_3 = np.array(dataset_3['Label_Data'])

pyplot.figure(1)
pyplot.plot(Label_Data_1[-100:], label = 'labels')
pyplot.legend()
pyplot.savefig("Label_April_Test")

pyplot.figure(2)
pyplot.plot(Label_Data_2, label = 'labels')
pyplot.legend()
pyplot.savefig("Label_May_Test")

pyplot.figure(3)
pyplot.plot(Label_Data_3[-400:-200], label = 'labels')
pyplot.legend()
pyplot.savefig("Label_June_Test")

DATA_PLOT_1 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_4_0703_NWA_Weather_NOFFT_Hourly.csv"
DATA_PLOT_2 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_5_0703_NWA_Weather_NOFFT_Hourly.csv"
DATA_PLOT_3 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_6_0703_NWA_Weather_NOFFT_Hourly.csv"

dataset_1 = pd.read_csv(DATA_PLOT_1, header = 0)
dataset_2 = pd.read_csv(DATA_PLOT_2, header = 0)
dataset_3 = pd.read_csv(DATA_PLOT_3, header = 0)

Label_Data_1 = np.array(dataset_1['Label_Data'])
Label_Data_2 = np.array(dataset_2['Label_Data'])
Label_Data_3 = np.array(dataset_3['Label_Data'])

pyplot.figure(4)
pyplot.plot(Label_Data_1[-100:], label = 'labels')
pyplot.legend()
pyplot.savefig("Label_April_Train")

pyplot.figure(5)
pyplot.plot(Label_Data_2, label = 'labels')
pyplot.legend()
pyplot.savefig("Label_May_Train")

pyplot.figure(6)
pyplot.plot(Label_Data_3, label = 'labels')
pyplot.legend()
pyplot.savefig("Label_June_Train")
'''