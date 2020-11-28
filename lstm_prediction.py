from datetime import datetime
from numpy import concatenate
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.layers import Dropout

import tensorflow as tf

print("Hello")
TRAIN_DATA_URL_1 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_6_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_1 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_6_0703_NWA_Weather_NOFFT_Hourly.csv"

# we take as trian data the May data. Because it has 31 days -> as a test dataset we would miss info on 31st day so last 144 time bins
TRAIN_DATA_URL_2 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_5_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_2 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_5_0703_NWA_Weather_NOFFT_Hourly.csv"

# we will test on June (because it has 30 days)
TRAIN_DATA_URL_3 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_4_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_3 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_4_0703_NWA_Weather_NOFFT_Hourly.csv"

# only for test
TRAIN_DATA_URL_4 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_4_2015_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_4 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_4_2015_0703_NWA_Weather_NOFFT_Hourly.csv"

TRAIN_DATA_URL_5 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_5_2015_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_5 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_5_2015_0703_NWA_Weather_NOFFT_Hourly.csv"

TRAIN_DATA_URL_6 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Test_DF_XGBoost_6_2015_0703_NWA_Weather_NOFFT_Hourly.csv"
TEST_DATA_URL_6 = "C:/Users/Matteo/Desktop/Python_Programs/LaFi_Train_DF_XGBoost_6_2015_0703_NWA_Weather_NOFFT_Hourly.csv"

# Function to build our dataset suitable for the LSTM
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()

	# input sequence (t-n, ..., t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ..., t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	#drop rows with NaN values
	if dropnan:
		agg.dropna(inplace = True)
	return agg

# Load the train and test dataset
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

train_dataset = pd.concat([train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4, train_dataset_5, train_dataset_6]) #I should use dataset_3 ONLY as test. No train data regarding dataset_3
test_dataset = pd.concat([test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4, test_dataset_5])

print(train_dataset)

train_values = train_dataset.values
test_values = test_dataset.values
test_values_3 = test_dataset_6.values 

train_values = train_values.astype('float32')
test_values = test_values.astype('float32')
test_values_3 = test_values_3.astype('float32') ###

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_values)
test_scaled = scaler.fit_transform(test_values)
test_scaled_3 = scaler.fit_transform(test_values_3) ###

print(train_scaled)

# Specify the number of lag hours we want to train or LSTM on
n_hours = 24 #30 is working good

# Specify the number of features in the dataset
n_features = 10


train_reframed = series_to_supervised(train_scaled, n_hours, 1)
test_reframed = series_to_supervised(test_scaled, n_hours, 1)
test_reframed_3 = series_to_supervised(test_scaled_3, n_hours, 1) ###

print(train_reframed.head())

#train_reframed.drop(train_reframed.columns[[20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
#30, 31, 32, 33, 34, 35, 36, 37, 38]], axis = 1, inplace = True) #maybe doing the series_to_supervised I am duplicating things
#test_reframed.drop(test_reframed.columns[[20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
#30, 31, 32, 33, 34, 35, 36, 37, 38]], axis = 1, inplace = True)

print(train_reframed.head())

train = train_reframed.values
test = test_reframed.values
test_3 = test_reframed_3.values ###

# Split into inputs (independent variables) and outputs (target values)
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
test_X_3, test_y_3 = test_3[:, :n_obs], test_3[:, -n_features] ###

print("This is train_y")
print(train_y[:100])

print(train_X.shape, len(train_X), train_y.shape)

# Reshape the dataset as requested by Keras
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
test_X_3 = test_X_3.reshape((test_X_3.shape[0], n_hours, n_features)) ###

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Build LSTM model architecture
model = Sequential()
model.add(LSTM(100, input_shape = (train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss = 'mae',
			optimizer = 'adam',
			metrics = ['mae', 'mse'])

EPOCHS = 100

# Train the model
history = model.fit(
	train_X,
	train_y,
	epochs = EPOCHS,
	batch_size = 128,
	validation_data = (test_X, test_y),
	verbose = 2,
	shuffle = False) # shuffle False is saying that samples in the bathes has to taken sequentially.
					 # Taking them randomly would not make sense when working with time series

# Plot history
pyplot.figure(0)
pyplot.plot(history.history['loss'], color = 'k', label='train loss')
pyplot.plot(history.history['val_loss'], color = 'r', label = 'validation loss')
pyplot.grid()
pyplot.ylabel('Mean Absolute Error (MAE)')
pyplot.xlabel('Number of training epochs')
pyplot.title('Training and validation loss value per number of epochs')
pyplot.legend()
pyplot.savefig('figura_dellastoria_80', dpi = 300)

#######################
# Make prediction
yhat_3 = model.predict(test_X_3) # the prediction for testing is done on test_3 dataset
test_X_3 = test_X_3.reshape((test_X_3.shape[0], n_hours*n_features)) ###
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

#######################
# Invert scaling for forecast on TEST data (I rescale the predicted output with the data on test_3)
inv_yhat_3 = concatenate((yhat_3, test_X_3[:, -9:]), axis = 1)
inv_yhat_3 = scaler.inverse_transform(inv_yhat_3)
inv_yhat_3 = inv_yhat_3[:,0]

print("len of in_hat_3 is {}".format(len(inv_yhat_3)))

# Invert scaling for validation data prediction
inv_yhat = concatenate((yhat, test_X[:, -9:]), axis = 1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

########################
# Invert scaling for actual (I take the real labels form test_3 dataset)
test_y_3 = test_y_3.reshape((len(test_y_3), 1)) ###
inv_y_3 = concatenate((test_y_3, test_X_3[:, -9:]), axis = 1) ###
inv_y_3 = scaler.inverse_transform(inv_y_3)
inv_y_3 = inv_y_3[:,0]

test_y = test_y.reshape((len(test_y), 1)) ###
inv_y = concatenate((test_y, test_X[:, -9:]), axis = 1) ###
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

#######################

# Plot some of the clusters predictions to visualize prediction performance

x_axis = np.zeros(100)
x_axis = np.linspace(24, 124, 100)

pyplot.figure(1) #before it was for cluster 2
pyplot.plot(x_axis, inv_yhat_3[(500+0*500):((500+0*500)+100)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y_3[(500+0*500):((500+0*500)+100)], color = 'b', marker = 'o',  markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluster 2 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Result_of_LSTM_Prediction_vs_TrueData_Cluster2_HQ", dpi = 300)

x_axis = np.zeros(100)
x_axis = np.linspace(24, 124, 100)

pyplot.figure(2) #before it was for cluster 2
pyplot.plot(x_axis, inv_yhat_3[(500+2*500):((500+2*500)+100)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y_3[(500+2*500):((500+2*500)+100)], color = 'b', marker = 'o',  markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluter 4 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Result_of_LSTM_Prediction_vs_TrueData_Cluster4_HQ", dpi = 300)

x_axis = np.zeros(100)
x_axis = np.linspace(24, 124, 100)

pyplot.figure(3)
pyplot.plot(x_axis, inv_yhat_3[(500+14*500):((500+14*500)+100)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y_3[(500+14*500):((500+14*500)+100)], color = 'b', marker = 'o', markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluster 16 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Result_of_LSTM_Prediction_vs_TrueData_Cluster16_HQ", dpi = 300)

x_axis = np.zeros(100)
x_axis = np.linspace(24, 124, 100)

pyplot.figure(4)
pyplot.plot(x_axis, inv_yhat_3[(500+26*500):((500+26*500)+100)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y_3[(500+26*500):((500+26*500)+100)], color = 'b', marker = 'o', markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluster 28 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Result_of_LSTM_Prediction_vs_TrueData_Cluster28_HQ", dpi = 300)

x_axis = np.zeros(100)
x_axis = np.linspace(24, 124, 100)

pyplot.figure(5)
pyplot.plot(x_axis, inv_yhat_3[(500+27*500):((500+27*500)+100)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y_3[(500+27*500):((500+27*500)+100)], color = 'b', marker = 'o', markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluster 29 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Cluster29_1_HQ", dpi = 300)

x_axis = np.zeros(100)
x_axis = np.linspace(124, 224, 100)

pyplot.figure(6)
pyplot.plot(x_axis, inv_yhat_3[((500+27*500)+100):((500+27*500)+200)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y_3[((500+27*500)+100):((500+27*500)+200)], color = 'b', marker = 'o', markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluster 29 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Cluster29_2_HQ", dpi = 300)

x_axis = np.zeros(100)
x_axis = np.linspace(224, 324, 100)

pyplot.figure(7)
pyplot.plot(x_axis, inv_yhat_3[((500+27*500)+200):((500+27*500)+300)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y_3[((500+27*500)+200):((500+27*500)+300)], color = 'b', marker = 'o', markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluster 29 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Cluster29_3_HQ", dpi = 300)

x_axis = np.zeros(100)
x_axis = np.linspace(324, 424, 100)

pyplot.figure(8)
pyplot.plot(x_axis, inv_yhat_3[((500+27*500)+300):((500+27*500)+400)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y_3[((500+27*500)+300):((500+27*500)+400)], color = 'b', marker = 'o', markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluster 29 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Cluster29_4_HQ", dpi = 300)

x_axis = np.zeros(100)
x_axis = np.linspace(424, 524, 100)

pyplot.figure(9)
pyplot.plot(x_axis, inv_yhat_3[((500+27*500)+400):((500+27*500)+500)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y_3[((500+27*500)+400):((500+27*500)+500)], color = 'b', marker = 'o', markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluster 29 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Cluster29_5_HQ", dpi = 300)

x_axis = np.zeros(100)
x_axis = np.linspace(24, 124, 100)

pyplot.figure(10)
pyplot.plot(x_axis, inv_yhat[(24):(124)], color = 'r', linestyle = 'dashed', marker = 'o', markersize = '4', label = 'Predicted (LSTM NN)')
pyplot.plot(x_axis, inv_y[(24):(124)], color = 'b', marker = 'o', markersize = '4', label = 'Real data')
pyplot.grid()
pyplot.xlabel('Time bin (hours)')
pyplot.ylabel('Number of pickups')
pyplot.title("Cluster 1 - Pickups per Hour: Real Data vs Prediction with LSTM NN")
pyplot.legend()
pyplot.savefig("Cluster1_HQ", dpi = 300)

mae_3 = mean_absolute_error(inv_y_3, inv_yhat_3)
mse_3 = mean_squared_error(inv_y_3, inv_yhat_3)

mae = mean_absolute_error(inv_y, inv_yhat)
mse = mean_squared_error(inv_y, inv_yhat)

print('Test MAE_3: %.3f' % mae_3)
print('Test MSE_3: %.3f' % mse_3)

print('Test MAE: %.3f' % mae)
print('Test MSE: %.3f' % mse)

# COMPUTATION OF MAPE

def s_mean_absolute_percentage_error(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred + 1)))*100 # mean on axis = -1

#mape = mean_absolute_percentage_error(inv_y[:-1], inv_yhat[1:])
#mape_3 = mean_absolute_percentage_error(inv_y_3[:-1], inv_yhat_3[1:])

mape = s_mean_absolute_percentage_error(inv_y, inv_yhat)
mape_3 = s_mean_absolute_percentage_error(inv_y_3, inv_yhat_3)

print('Test MAPE: %.3f' % mape)
print('Test MAPE_3: %.3f' % mape_3)

mape_special_200_400 = s_mean_absolute_percentage_error(inv_y_3[200:400], inv_yhat_3[200:400])
print('Test MAPE_SPECIAL_200_400: %.3f' % mape_special_200_400)
mape_special_800_1000 = s_mean_absolute_percentage_error(inv_y_3[800:1000], inv_yhat_3[800:1000])
print('Test MAPE_special_800_1000: %.3f' % mape_special_800_1000)
mape_special_1200_1400 = s_mean_absolute_percentage_error(inv_y_3[1200:1400], inv_yhat_3[1200:1400])
print('Test MAPE_special_800_1000: %.3f' % mape_special_1200_1400)

print("Number of input for test is = {}".format(len(test_X_3[:10])))
print(inv_yhat_3[:10])
print(inv_yhat_3[10])
print(inv_yhat_3[9])
print("Number of outputs for test is = {}".format(len(inv_yhat_3)))

# Regional sMAPE (per each Cluster)
mape = s_mean_absolute_percentage_error(inv_y_3[24:450], inv_yhat_3[24:450])
print('Test sMAPE Cluster1: %.3f' % mape)

# +500 includes the borders as well: try to exclude them. Let's see how this prediction goes and decide

for i in range(0,29): # starts from 0 but I am actually computing starting from cluster n.1
	mape = s_mean_absolute_percentage_error(inv_y_3[((500+i*500)+24):((500+i*500)+450)], inv_yhat_3[((500+i*500)+24):((500+i*500)+450)])
	print('Test sMAPE Cluster{}: {}'.format((i+2), mape))

# Regional RMSE (per each Cluster)
rmse = math.sqrt(mean_squared_error(inv_y_3[24:450], inv_yhat_3[24:450]))
print('Test RMSE Cluster1: %.3f' % rmse)

for i in range(0,29): # starts from 0 but I am actually computing starting from cluster n.1
	rmse = math.sqrt(mean_squared_error(inv_y_3[(500+i*500)+24:((500+i*500)+450)], inv_yhat_3[(500+i*500)+24:((500+i*500)+450)]))
	print('Test RMSE Cluster{}: {}'.format((i+2), rmse))