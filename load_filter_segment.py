# This code is in great part inspired from examples I found on the web to treat taxi ride data

import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import folium
from datetime import datetime
import time
import seaborn as sns
import os
import math
import gpxpy.geo
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load data
data_2014 = dd.read_csv("C:/Users/Matteo/Desktop/University/Second_Year/Papers/OK_4/yellow_tripdata_2016-04.csv", sep=',')
print(data_2014.head()) # to print the head of the file that I saved in the structure data_2014

data_2014.columns
#Index(['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'pickup_longitude',
#	'pickup_latitude', 'RateCodeID', 'store_and_fwd_flag', 'dropoff_longitude', 'dropoff_latitude', 'payment_type',
#	'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount'], dtype='object')

print("Number of columns = "+str(len(data_2014.columns)))

# Plot pickups outside of NYC borders
outside_NYC = data_2014[((data_2014.pickup_latitude <= 40.5774) | (data_2014.pickup_longitude <= -74.15) | (data_2014.pickup_latitude >= 40.9176) | (data_2014.pickup_longitude >= -73.7004))]

m = folium.Map(location = [40.5774, -73.7004], tiles = "Stamen Toner")

outside_pickups = outside_NYC.head(5400)

for i,j in outside_pickups.iterrows():
	if j["pickup_latitude"] != 0:
		folium.Marker([j["pickup_latitude"], j["pickup_longitude"]]).add_to(m)

m.save("pickup_outside_NYC.html")

# Plot pickups outside of NYC borders
outside_NYC = data_2014[((data_2014.dropoff_latitude <= 40.5774) | (data_2014.dropoff_longitude <= -74.15) | (data_2014.dropoff_latitude >= 40.9176) | (data_2014.dropoff_longitude >= -73.7004))]

m = folium.Map(location = [40.5774, -73.7004], tiles = "Stamen Toner")

outside_dropoffs = outside_NYC.head(5400)

for i,j in outside_dropoffs.iterrows():
	if j["dropoff_latitude"] != 0:
		folium.Marker([j["dropoff_latitude"], j["dropoff_longitude"]]).add_to(m)

m.save("dropoff_outside_NYC.html")

# Function to convert time from 'Y-m-d H:M:S' to 'Unix Time' format
def timeToUnix(t):

	change = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
	t_tuple = change.timetuple()
	return time.mktime(t_tuple) - 25200 # 28800 if -8 from GMT, 25200 if -7 from GMT

# Function to add features to the dataset
def dfWithTripTimes(df):
	startTime = datetime.now()
	duration = df[["tpep_pickup_datetime", "tpep_dropoff_datetime"]].compute()
	pickup_time = [timeToUnix(pkup) for pkup in duration["tpep_pickup_datetime"].values]
	dropoff_time = [timeToUnix(drpof) for drpof in duration["tpep_dropoff_datetime"].values]

	trip_duration = (np.array(dropoff_time) - np.array(pickup_time))/float(60)

	NewFrame = df[['passenger_count','trip_distance','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','total_amount']].compute()
	NewFrame["trip_duration"] = trip_duration
	NewFrame["pickup_time"] = pickup_time
	NewFrame["speed"] = (NewFrame["trip_distance"]/NewFrame["trip_duration"])*60

	print("Time taken for creation of dataframe is {}".format(datetime.now()-startTime))
	return NewFrame

new_frame = dfWithTripTimes(data_2014)

# Filter by TRIP DURATIONS
plt.figure(figsize = (10,6))
sns.boxplot("trip_duration", data = new_frame, orient = "v")
plt.tick_params(labelsize = 20)
plt.ylabel("Trip Duration(minutes)", fontsize = 20)
plt.savefig('figure1.png')

quantile_tripDuration = new_frame.trip_duration.quantile(np.round(np.arange(0.00, 1.01, 0.01),2)) # ???

qValues = np.round(np.arange(0.00, 1.01, 0.1), 2) # what are np.round and np.range
for i in qValues:
	print("{}th percentile value of Trip Duration is {}min".format((int(i*100)), quantile_tripDuration[i]))

qValues = np.round(np.arange(0.9, 1.01, 0.01), 2)
for i in qValues:
	print("{} percentile value of Trip Duration is {}min".format((int(i*100)), quantile_tripDuration[i]))

new_frame_cleaned = new_frame[(new_frame.trip_duration > 1) & (new_frame.trip_duration<720)]

# Trip duration after cleaning the data that had too long or negative trip duration
plt.figure(figsize = (10,6))
sns.boxplot("trip_duration", data = new_frame_cleaned, orient = "v")
plt.ylim(ymin = 1, ymax = 750)
plt.tick_params(labelsize = 20)
plt.ylabel("Trip Duration(minutes)", fontsize = 20)
plt.savefig('figure2.png')

# PDF of the trip duration
plt.figure(figsize = (12,8))
sns.kdeplot(new_frame_cleaned["trip_duration"].values, shade = True, cumulative = False)
plt.tick_params(labelsize = 20)
plt.xlabel("Trip Duration", fontsize = 20)
plt.title("PDF of Trip Duration", fontsize = 20)
plt.savefig('figure3.png')

# Filter by SPEED of the trips

def changingLabels(num):
	if num < 10**3:
		return num
	elif num >= 10**3 and num < 10**6:
		return str(num/10**3) + "k"
	elif num >= 10**6 and num < 10**9:
		return str(num/10**6) + "M"
	else:
		return str(num/10**9) + "B"

fig = plt.figure(figsize = (10,6))
ax = sns.boxplot("speed", data = new_frame_cleaned, orient = "v")

ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
plt.tick_params(labelsize = 20)
plt.ylabel("Speed(Miles/hr) in Millions", fontsize = 20)
plt.savefig('figure4.png')

quantile_speed = new_frame_cleaned.speed.quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))

qValues = np.round(np.arange(0.00, 1.01, 0.1), 4)
for i in qValues:
	print("{}th percentile value of speed is {}miles/hr".format(int(i*100), quantile_speed[i]))

qValues = np.round(np.arange(0.91, 1.01, 0.01), 3)
for i in qValues:
	print("{} percentile value of speed is {}miles/hr".format(int(i*100), quantile_speed[i]))

qValues = np.round(np.arange(0.991, 1.001, 0.001), 4)
quantile_speed  = new_frame_cleaned.speed.quantile(qValues)
for i in qValues:
	print("{} percentile value of speed is {}miles/hr".format((i*100), quantile_speed[i]))

new_frame_cleaned = new_frame_cleaned[(new_frame_cleaned.speed > 0) & (new_frame_cleaned.speed < 45.43)]

# plot of speed after removing the outliers and erronedous points
fig = plt.figure(figsize = (10,6))
ax = sns.boxplot("speed", data = new_frame_cleaned, orient = 'v')

plt.tick_params(labelsize = 20)
plt.ylabel("Speed(Miles/hr)", fontsize = 20)
plt.savefig('figure5.png')

Average_speed = sum(new_frame_cleaned.speed)/len(new_frame_cleaned.speed)
print("Average Speed of Taxis around NYC = "+str(Average_speed))

print("Speed of Taxis around NYC per 10 minutes = "+str(Average_speed/6)+" per 10 minutes") # ??? That tells how many miles can a taxi travel every 10 minutes

# Filter by TRIP DISTANCE

fig = plt.figure(figsize = (10,6))
ax = sns.boxplot("trip_distance", data = new_frame_cleaned, orient = "v")

plt.tick_params(labelsize = 20)
plt.ylabel("Trip Distance(Miles)", fontsize = 20)
plt.savefig('figure6.png')

quantile_tripDistance = new_frame_cleaned.trip_distance.quantile(np.around(np.arange(0.00, 1.01, 0.01), 2))

qValues = np.round(np.arange(0.00, 1.01, 0.1), 2)
for i in qValues:
	print("{}th percentile value of trip distance is {}miles".format(int(i*100), quantile_tripDistance[i]))

qValues = np.round(np.arange(0.91, 1.01, 0.01), 3)
for i in qValues:
	print("{} percentile value of trip distance is {}miles".format(int(i*100), quantile_tripDistance[i]))

quantile_tripDistance = new_frame_cleaned.trip_distance.quantile(np.round(np.arange(0.991, 1.001, 0.001), 4))
qValues = np.round(np.arange(0.991, 1.001, 0.001), 3)
for i in qValues:
	print("{} percentile vlaue of trip distance is {}miles".format((i*100), quantile_tripDistance[i]))

new_frame_cleaned = new_frame_cleaned[(new_frame_cleaned.trip_distance>0) & (new_frame_cleaned.trip_distance<23)] #page 24 of guide

#plot of tirp distance after removing outlier points
fig = plt.figure(figsize = (10,6))
ax = sns.boxplot("trip_distance", data = new_frame_cleaned, orient = "v")

plt.tick_params(labelsize = 20)
plt.ylabel("Trips Distance(Miles)", fontsize = 20)
plt.savefig('figure7.png')

# Filter by TOTAL FARE of the rides

fgu = plt.figure(figsize = (10,6))
ax = sns.boxplot("total_amount", data = new_frame_cleaned, orient = "v")

plt.tick_params(labelsize = 20)
plt.ylabel("Trip Fare", fontsize = 20)
plt.savefig('figure8.png')

quantile_totalAmount = new_frame_cleaned.total_amount.quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))

qValues = np.round(np.arange(0.00, 1.01, 0.1), 2)
for i in qValues:
	print("{}th percentile value of trips fare is {}".format(int(i*100), quantile_totalAmount[i]))

qValues = np.round(np.arange(0.91, 1.01, 0.01), 3)
for i in qValues:
	print("{} percentile value of trip fare is {}".format(int(i*100), quantile_totalAmount[i]))

quantile_totalAmount = new_frame_cleaned.total_amount.quantile(np.round(np.arange(0.991, 1.001, 0.001), 3))
qValues = np.round(np.arange(0.991, 1.001, 0.001), 3)
for i in qValues:
	print("{} percentile value of trip fare is {}".format((i*100), quantile_totalAmount[i]))

new_frame_cleaned = new_frame_cleaned[(new_frame_cleaned.total_amount>0) & (new_frame_cleaned.total_amount<88.0)]

#plot of fare amount after removing outliers and erroneous points
fig = plt.figure(figsize = (10,6))
ax = sns.boxplot("total_amount", data = new_frame_cleaned, orient = "v")

plt.tick_params(labelsize = 20)
plt.ylabel("Trip Fare", fontsize = 20)
plt.savefig('figure9.png')

# Filter REMOVING rides with PICKUPS OUTSIDE NYC

new_frame_cleaned = new_frame_cleaned[(((new_frame_cleaned.pickup_latitude >= 40.5774) & (new_frame_cleaned.pickup_latitude <= 40.9176))
										& ((new_frame_cleaned.pickup_longitude >= -74.15) & (new_frame_cleaned.pickup_longitude <= -73.7004)))]

m = folium.Map(location = [40.9176, -73.7004], tiles = "Stamen Toner")

pickups_within_NYC = new_frame_cleaned.sample(n = 500) # taking just 500 elements ???

for i,j in pickups_within_NYC.iterrows():
	folium.Marker([j["pickup_latitude"], j["pickup_longitude"]]).add_to(m)

m.save("pickup_within_NYC.html")

# Filter REMOVING rides with DROPOFFS OUTSIDE NYC

new_frame_cleaned = new_frame_cleaned[(((new_frame_cleaned.dropoff_latitude >= 40.5774) & (new_frame_cleaned.dropoff_latitude <= 40.9176))
										& ((new_frame_cleaned.dropoff_longitude >= -74.15) & (new_frame_cleaned.dropoff_longitude <= -73.7004)))]

m = folium.Map(location = [40.9176, -73.7004], tiles = "Stamen Toner")

dropoffs_within_NYC = new_frame_cleaned.sample(n = 500) # taking just 500 elements ???

for i,j in dropoffs_within_NYC.iterrows():
	folium.Marker([j["dropoff_latitude"], j["dropoff_longitude"]]).add_to(m)

m.save("dropoff_within_NYC.html")

# Clustering/Segmentation

coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]].values
neighbors = []

# Function to compute minimum distance between clusters
def min_distance(regionCenters, totalClusters):
	good_points = 0
	bad_points = 0
	less_dist = []
	more_dist = []
	min_distance = 100000
	for i in range (totalClusters):
		good_points = 0
		bad_points = 0
		for j in range(totalClusters):
			if j != i:
				distance = gpxpy.geo.haversine_distance(latitude_1 = regionCenters[i][0],
					longitude_1 = regionCenters[i][1], latitude_2 = regionCenters[j][0], longitude_2 = regionCenters[j][1])
				distance = distance/(1.60934*1000) #conversion from km to miles
				min_distance = min(min_distance, distance)

				if distance < 2:
					good_points += 1
				else:
					bad_points += 1
		less_dist.append(good_points)
		more_dist.append(bad_points)
	print("On choosing a cluster size of {}".format(totalClusters))
	print("Avg. Number clusters within vicinity where inter cluster distance < 2 miles is {}".format(np.ceil(sum(less_dist)/len(less_dist))))
	print("Avg. Number clusters outside of vicinity where inter cluster distance > 2 miles is {}".format(np.ceil(sum(more_dist)/len(more_dist))))
	print("Minimum distance between any two clusters = {}".format(min_distance))
	print("-"*10)

# Function to generate the clusters with MiniBatchKMeans function from Scikit-learn library
def makingRegions(noOfRegions):
	regions = MiniBatchKMeans(n_clusters = noOfRegions, batch_size = 10000).fit(coord)
	regionCenters = regions.cluster_centers_
	totalClusters = len(regionCenters)
	return regionCenters, totalClusters

# Try different numbers of clusters to see what is the minimum distance between clusters
startTime = datetime.now()
for i in range(10, 100, 10): # i will range from 10 to 100 with step 10 -> to try different numbers of regions
	regionCenters, totalClusters = makingRegions(i) # create the regions, with center and elements inside
	min_distance(regionCenters, totalClusters) # computes the minimum distance between regions
print("Time taken = "+str(datetime.now() - startTime))

# now we choose a number of clusters knowing that we want the minimum distance to be smaller than 0.5

NUM_CLUSTERS = 30;

coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]].values
regions = MiniBatchKMeans(n_clusters = NUM_CLUSTERS, batch_size = 10000).fit(coord)
new_frame_cleaned["pickup_cluster"] = regions.predict(new_frame_cleaned[["pickup_latitude", "pickup_longitude"]])

# it is possible that using MiniBatchKMeans we introduce some radomness into the algorithm so that the location of the
# centers of the regions change a bit, and so the number of time bins with zero pickups per region is changing form
# one instance to another

print(new_frame_cleaned.head())

# Plot the centers of the regions

centerOfRegions = regions.cluster_centers_
noOfClusters = len(centerOfRegions)
m = folium.Map(location = [40.9176, -73.7004], tiles = "Stamen Toner")

for i in range(noOfClusters):
	folium.Marker([centerOfRegions[i][0], centerOfRegions[i][1]], popup = (str(np.round(centerOfRegions[i][0], 2))+", "
		+str(np.round(centerOfRegions[i][1], 2)))).add_to(m)
m.save("Cluster_Centers_Map.html")

# Plot the regions in NYC

NUM_ELEMENTS = 7000
NYC_Latitude_range = (40.5774, 40.9176)
NYC_Longitude_range = (-74.15, -73.7004)
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,1.5])
ax.scatter(x = new_frame_cleaned.pickup_longitude.values[:NUM_ELEMENTS], y = new_frame_cleaned.pickup_latitude.values[:NUM_ELEMENTS],
	c = new_frame_cleaned.pickup_cluster.values[:NUM_ELEMENTS], cmap = "Paired", s = 5)
ax.set_xlim(-74.10, -73.72)
ax.set_ylim(40.6774, 40.9176)
ax.set_title("Regions in New York City")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.savefig('figure10.png')

# 1397433600 unix time equivalent for 4/14/2014 12:00:00 AM GMT
# 1397645483 unix time equivalent for 4/16/2014 10:51:23 AM GMT

# Time slot segmentation
def pickup_1h_bins(dataframe, month, year):
	pickupTime = dataframe["pickup_time"].values
	unixTime = [1459468800] # time at 4/1/2016 00:00:00 || March = 1456790400,  #2016  For April: 1459468800, For May: 1462060800, For June: 1464739200
																				#2015  For April: 1427846400, For May: 1430438400, For June: 1433116800
	unix_year = unixTime[year-2016] # move to hear 2015 for 2015
	time_1hr_bin = [int((i - unix_year)/3600) for i in pickupTime] # 3600 is the UnixTime of 1 hour
	dataframe["time_bin"] = np.array(time_1hr_bin)
	return dataframe

apr_2014_data = pickup_1h_bins(new_frame_cleaned, 4, 2016)
print(apr_2014_data.head())

print("There should be ... unique 10 minute time bins for the selected days of April 2014: ", str(len(np.unique(apr_2014_data["time_bin"]))))

apr_2014_timeBin_groupBy = apr_2014_data[["pickup_cluster", "time_bin", "trip_distance"]].groupby(by = ["pickup_cluster", "time_bin"]).count()

print(apr_2014_timeBin_groupBy.head())

print("The number of cleaned data is : {}".format(len(new_frame) - len(new_frame_cleaned)))
print("The ratio between original set and data in use is: {}".format(len(new_frame_cleaned)/len(new_frame)))

# SMOOTHING

# get the unique time bins where pickup are present for each region

NUM_BINS = 720 # for 30 days = 4320, for 31 days = 4464. ||| 720 for 30 days. 744 for 31 days

def getUniqueBinsWithPickups(dataframe):
	values = []
	for i in range(NUM_CLUSTERS):
		cluster_id = dataframe[dataframe["pickup_cluster"] == i] # here it takes elements of the dataframe that has "i" as cluster
		unique_clus_id = list(set(cluster_id["time_bin"])) # here I create a list with the time bins. set() is counting only once the bins that are repeating
		unique_clus_id.sort() # sort so that to sort the bins IDs
		values.append(unique_clus_id)
	return values
# this function is returning the indices of all the unique time_bins where THERE IS a pickup for all the 30 clusters

# now we want to find out how many bins per cluster are empty (no pickups)
unique_binswithPickup_apr_2014 = getUniqueBinsWithPickups(apr_2014_data)
for i in range(NUM_CLUSTERS): # 30 clusters, 353 time bins
	print("For cluster ID {}, total number of time bins with no pickup in this cluster region is {}".format(i,
		(NUM_BINS - len(unique_binswithPickup_apr_2014[i])))) # len(.) is saying how many time bins per cluster we have
	print("-"*90)

# following we fill the gaps with 0

def fillMissingWithZero(numberOfPickups, correspondingTimeBin): #number of pickups comes form the GroupBy object
	ind = 0
	smoothed_regions = []
	for c in range(0, NUM_CLUSTERS):
		smoothed_bins = []
		for t in range(NUM_BINS):
			if t in correspondingTimeBin[c]: # correspondingTimeBin has for each cluster the number of time bins where there is at least one pickup
				# if the t-th time bin belongs to the cluster c-th
				smoothed_bins.append(numberOfPickups[ind]) #number of pickups at a certain time bin at a certain cluster
				ind += 1 # only if the bin belongs to the cluster it makes sense to increase ind, otherwise the bin is not present
			else:
				smoothed_bins.append(0)
		smoothed_regions.extend(smoothed_bins)
	return smoothed_regions # it return an objects with 30 clusters and for each cluster 354 time bins

# following we fill the gaps with the avarage of the neighbors points

# in correspondingTimeBin I have for each cluster the existing time bins (time bins where there are actually pickups)
# in numberOfPickups I have the info of number of pikcups for that pickup
def smoothing(numberOfPickups, correspondingTimeBin):
	ind = 0
	repeat = 0
	smoothed_region = []
	for cluster in range(0, NUM_CLUSTERS): # for each cluster
		smoothed_bin = []
		for t1 in range(NUM_BINS):
			if repeat != 0:
				repeat -= 1 # to skip the already treated TimeBins to save time
			else:
				# if t1 is a TimeBin of the that cluster (so if there are pickups in this TimeBin)
				if t1 in correspondingTimeBin[cluster]: # the unique time bins vector relative to the specific cluster
					smoothed_bin.append(numberOfPickups[ind])
					ind += 1 # keep tracks of what indices we have in the actual numberOfPickups
				else:
					if t1 == 0: # if it is the first TimeBin of the cluster
#<--------------------CASE 1: The pickups are missing at the beginning ---------------------------------->
						for t2 in range(t1, NUM_BINS): # looking for TimeBins from t1 to the total number of TimeBins
							if t2 not in correspondingTimeBin[cluster]: # When t2 is not in the TimeBins of the cluster
								continue # goes again to 'for t2 in range()'
							else: # When we reached the end of empty TimeBins and we have a TimeBin with pickups
								right_hand_limit = t2
								smoothed_value = (numberOfPickups[ind]*1.0)/((right_hand_limit + 1)*1.0) # once we have the right_hand_limit we can compute the average. right_hand_limit coincide with the count of steps when we are missing data at the beginning
								for i in range(right_hand_limit + 1): # Here I assign the smoothed_value to the missing time bins
									smoothed_bin.append(math.ceil(smoothed_value))
								ind += 1
								repeat = right_hand_limit - t1
					if t1 != 0: # if t1 is not the first TimeBin of the cluster (it could be in the middle or at the end)
						right_hand_limit = 0 # initialized so that in the case the set has no right limit, we take it as zero.
						for t2 in range(t1, NUM_BINS):
							if t2 not in correspondingTimeBin[cluster]:
								continue # when we don't find the TimeBin in the time bins corresponding to the cluster
							else:
								right_hand_limit = t2
								break # when we find the right_hand_limit we no longer search for it
						if right_hand_limit == 0: # this means that we didn't find a right_hand_limit
#<--------------------CASE 2: The pickups are missing at the end ---------------------------------->
							smoothed_value = (numberOfPickups[ind-1]*1.0)/(((NUM_BINS - t1)+1)*1.0) # we compute the smoothed value
							del smoothed_bin[-1] # we delete it beacuse we are going to substitute it too, with the smoothed value
							for i in range((NUM_BINS - t1)+1):
								smoothed_bin.append(math.ceil(smoothed_value))
							repeat = (NUM_BINS - t1) - 1 # to skip already treated bins
#<--------------------CASE 3: The pickups are missing in the middle ---------------------------------->
						else:
							smoothed_value = ((numberOfPickups[ind-1] + numberOfPickups[ind])*1.0)/(((right_hand_limit - t1)+2)*1.0)
							del smoothed_bin[-1] # we delete it beacuse we are going to substitute it with the smoothed value
							for i in range((right_hand_limit - t1)+2):
								smoothed_bin.append(math.ceil(smoothed_value))
							ind += 1
							repeat = right_hand_limit - t1
		smoothed_region.extend(smoothed_bin)
	return smoothed_region

# unique_binswithPickups_apr_2014 says per each cluster, how many bins there are that have at least one pickup
apr_2014_fillZero = fillMissingWithZero(apr_2014_timeBin_groupBy["trip_distance"].values, unique_binswithPickup_apr_2014)
apr_2014_fillSmooth = smoothing(apr_2014_timeBin_groupBy["trip_distance"].values, unique_binswithPickup_apr_2014)

def countZeros(num): # to count the zeros withing the smoothed data
		count = 0
		for i in num:
			if i == 0:
				count += 1
		return count

print("Number of values filled with zero in zero fill data = "+str(countZeros(apr_2014_fillZero)))
print("Sanity check for number of zeros in smoothed data = "+str(countZeros(apr_2014_fillSmooth)))

print("Total number of pickup values = "+str(len(apr_2014_fillSmooth)))

# In the next, we write [NUM_BINS*(n-1):NUM_BINS*n] because apr_2014_fillZero is a single vector that every each NUM_BINS
# elements it has information about a new cluster

fig = plt.figure(figsize =(18, 10))
plt.plot(apr_2014_fillZero[NUM_BINS*0:NUM_BINS*29], label = "Filled With Zero")
plt.plot(apr_2014_fillSmooth[NUM_BINS*0:NUM_BINS*29], label = "Filled with Avg Values (Smooth)")
plt.legend(bbox_to_anchor = (1, 1), fontsize = 18)
#plt.ylim(0,20)
#plt.xlim(0,4400)
plt.tick_params(labelsize = 20)
plt.savefig('figure11.png')

# region wise data building after the smoothing operation
regionWisePickup_apr_2014 = []
for i in range(NUM_CLUSTERS):
	regionWisePickup_apr_2014.append(apr_2014_fillSmooth[(NUM_BINS*i):((NUM_BINS*i)+NUM_BINS)])
# Here I append in a list of lists all the bins for all the cluster.
# I have as many rows as the number of clusters and as second dimension the number of time bins


# regionWisePickup_apr_2014 is a dataset that contains clusters on rows, bins on columns,
# and as entrues contains the number of pickups for the specific cluster-bin couple
print("Number of clusters on the data strucutre is: {}".format(len(regionWisePickup_apr_2014)))
print("Number of time bins per cluster in the data structure is: {}".format(len(regionWisePickup_apr_2014[0])))

# ADD WEATHER (TEMPERATURE) DATA

# Importing weather data and building weather info numpy arrays

DATA_URL = "C:/Users/Matteo/Desktop/University/Second_Year/Papers/OK_4/Weather_NYC_Hourly_4_2016.csv"

weather_dataset = pd.read_csv(DATA_URL, header = 0, sep = ';')

print(weather_dataset.head())

print("Number of columns = "+str(len(weather_dataset.columns)))

print(weather_dataset["Temperature"].head()) # print the columns relative to the temperature

weather_dataset["Temperature"] = weather_dataset["Temperature"].str.replace('F', '')
weather_dataset["Temperature"] = weather_dataset["Temperature"].astype('float32')

#weather_dataset["Pressure"] = weather_dataset["Pressure"].str.replace('in', '')
#weather_dataset["Pressure"] = weather_dataset["Pressure"].astype('float32')

print(weather_dataset["Temperature"].head())
#print(weather_dataset["Pressure"].head())

Hourly_Temperature = np.array(weather_dataset["Temperature"])
#Hourly_Pressure = np.array(weather_dataset["Pressure"])

# PREPARING THE DATASET FOR LSTM AND XGBOOST

# to take the number of pickups happened in the last 5 10min intervals
number_of_time_stamps = 5

# we are taking previous 5 pickups as a training data for predictiong next pickup
# Ground truth: that is the reality we want want model to predict

TruePickups = [] #it will not contain first 5 pickups of each cluster

lat = [] # will contain NUM_BIN-5 times latitude of clusters for every cluster

lon = [] # will contain NUM_BIN-5 times longitude of clusters for every cluster

# sunday = 0, monday = 1, tue = 3, wed = 4, thu = 5, fri = 6, sat = 7
day_of_week = [] # for each cluster we will add NUM_BIN-5 values representing the day of the week

temperature = []
#pressure = []

feat = []

centerOfRegions = regions.cluster_centers_
feat = [0]*number_of_time_stamps

# I am creating lists of lists: for each clusters I have info about each time bin: dimensions are: NUM_CLUSTERS*(NUM_BINS-number_of_time_stamps)
for i in range(NUM_CLUSTERS):
	lat.append([centerOfRegions[i][0]]*(NUM_BINS-number_of_time_stamps))
	lon.append([centerOfRegions[i][1]]*(NUM_BINS-number_of_time_stamps))
	# 144 is the number of 10mins bins in one day
	# the following assign a day of the week every tot time bins
	day_of_week.append([int(((int(j/24)%7)+5)%7) for j in range(5, NUM_BINS)]) # (144 would be 24 for hourly bins)
																			   # +5 (Fri) for April, +7 (Sun) for May, +3 (Wed) for June
																			   # 
	temperature.append([Hourly_Temperature[int(j)] for j in range(5, NUM_BINS)])
	#pressure.append([Hourly_Pressure[int(j/6)] for j in range(5, NUM_BINS)])
	# add frequencies:
	feat = np.vstack((feat, [regionWisePickup_apr_2014[i][k:k+number_of_time_stamps] # taking off the first 5 elements
		for k in range(0, len(regionWisePickup_apr_2014[i]) - (number_of_time_stamps))])) 
	TruePickups.append(regionWisePickup_apr_2014[i][5:]) # first 5 elements are not here cause they do not have 5 elements before them. THe first 5 elements are needed to predict the 6th and so on

feat = feat[1:] # skip the first element
# dim of feat is [(NUM_BINS-5) * NUM_CLUSTERS][5]
print(feat)

print("Length of day_of_week is = {}".format(len(day_of_week[0])))
print("Length of temperature is = {}".format(len(temperature[0])))
#print("Length of pressure is = {}".format(len(pressure[0])))

# the following condition is true when we have the correct number of elements in vectors
print("Data are collected in a good way: {}".format(len(lat[0])*len(lat) == len(lon[0])*len(lon) == len(day_of_week[0])*len(day_of_week) == (NUM_BINS-number_of_time_stamps)*NUM_CLUSTERS == len(TruePickups[0])*len(TruePickups)))

# We want to add also the weighted moving average prediction as a feature in our data
# (I tried both with and without the weighted average: performance don't change a lot)

predicted_pickup_values = []
predicted_pickup_values_list = []
predicted_value = -1 # -1 is only a default value

window_size = 2
for i in range(NUM_CLUSTERS):
	for j in range(NUM_BINS):
		if j == 0:
			predicted_value = regionWisePickup_apr_2014[i][j]
			predicted_pickup_values.append(0)
		else:
			if j >= window_size:
				sumPickups = 0
				sumOfWeights = 0
				for k in range(window_size, 0, -1):
					sumPickups += k*(regionWisePickup_apr_2014[i][j-window_size + (k-1)])
					sumOfWeights += k
				predicted_value = int(sumPickups/sumOfWeights)
				predicted_pickup_values.append(predicted_value)
			else:
				sumPickups = 0
				sumOfWeights = 0
				for k in range(j, 0, -1):
					sumPickups += k*regionWisePickup_apr_2014[i][k-1]
					sumOfWeights += k
				predicted_value = int(sumPickups/sumOfWeights)
				predicted_pickup_values.append(predicted_value)
	predicted_pickup_values_list.append(predicted_pickup_values[5:])
	predicted_pickup_values = []

print("Condition is {}".format(len(predicted_pickup_values_list[0])*len(predicted_pickup_values_list) == (NUM_BINS-number_of_time_stamps)*NUM_CLUSTERS))

# COMPUTE AND PLOT THE FOURIER TRANSFORM OF THE TIME SERIES FOR EACH CLUSTER

# Compute FFT info (amplitude of peaks and frequencies)
amplitude_lists = []
frequency_lists = []
for i in range(NUM_CLUSTERS):
	ampli = np.abs(np.fft.fft(regionWisePickup_apr_2014[i][0:NUM_BINS]))
	freq = np.abs(np.fft.fftfreq(NUM_BINS, 1)) # we concern only about positive frquencies
	ampli_indices = np.argsort(-ampli)[1:] # returns array of indeces for which correposnding amplitude values are sorted in reverse order
	amplitude_values = []
	frequency_values = []
	for j in range(0, 9, 2): # from 0 to 9 with step 2 -> we are taking the top five amplitudes and frequencies
		amplitude_values.append(ampli[ampli_indices[j]]) # why step 2: because the fourier transform is a repeating wave
														 # the fourier transform is mirrored
		frequency_values.append(freq[ampli_indices[j]])
	for k in range(NUM_BINS-number_of_time_stamps):
		amplitude_lists.append(amplitude_values) # each 'amplitude_values' has the 5 peaks and we are appending for NUM_CLUSTERS*k times
		frequency_lists.append(frequency_values) # each 'frequency_values' has the 5 frquencies and we are appending for NUM_CLUSTERS*k times

# Now we have 19 features for the input data: lat, lon, day_of_week, weighted moving avg, 5 number of pickups,
# 5 main amplitudes of FFT, 5 corresponding freq of FFT

# DATA PREPARATION FOR REGRESSION MODELS

# For every region we want 80% of data for train, 20% of data for test

NUM_BINS_USED = NUM_BINS-number_of_time_stamps
NUM_BINS_TRAIN = int(NUM_BINS_USED*0.7)
NUM_BINS_TEST = int(NUM_BINS_USED*0.3)

# dividing between train and test set for pickups, frequencies and amplitudes

train_previousFive_pickups = [feat[i*NUM_BINS_USED:(NUM_BINS_USED*i+NUM_BINS_TRAIN)] for i in range(NUM_CLUSTERS)]
test_previousFive_pickups = [feat[(i*NUM_BINS_USED+NUM_BINS_TRAIN):(NUM_BINS_USED*(i+1))] for i in range(NUM_CLUSTERS)]

# train_previousFive_pickups and test_previousFive_pickups are tridimensional vector because for each cluster (first dimension)
# I take tot elements of feat (second dimension) and each row of feat has 5 elements (third dimension)
# The same hold for fourier frequencies and fourier amplitudes

train_fourier_frequencies = [frequency_lists[i*NUM_BINS_USED:(NUM_BINS_USED*i+NUM_BINS_TRAIN)] for i in range(NUM_CLUSTERS)]
test_fourier_frequencies = [frequency_lists[(i*NUM_BINS_USED+NUM_BINS_TRAIN):(NUM_BINS_USED*(i+1))] for i in range(NUM_CLUSTERS)]

train_fourier_amplitudes = [amplitude_lists[i*NUM_BINS_USED:(NUM_BINS_USED*i+NUM_BINS_TRAIN)] for i in range(NUM_CLUSTERS)]
test_fourier_amplitudes = [amplitude_lists[(i*NUM_BINS_USED+NUM_BINS_TRAIN):(NUM_BINS_USED*(i+1))] for i in range(NUM_CLUSTERS)]

print("Train Data: Total number of clusters = {}. Number of points in each cluster = {}. Total number of training points = {}"
	.format(len(train_previousFive_pickups), len(train_previousFive_pickups[0]), len(train_previousFive_pickups)*len(train_previousFive_pickups[0])))
print("Test Data: Total number of clusters = {}. Number of points in each cluster = {}. Total number of test points = {}"
	.format(len(test_previousFive_pickups), len(test_previousFive_pickups[0]), len(test_previousFive_pickups)*len(test_previousFive_pickups[0])))

# 80% + 20% = 100% = (NUM_BINS-number_of_time_stamps)

TOT_train_points = len(train_previousFive_pickups[0])
TOT_test_points = len(test_previousFive_pickups[0])

# Taking 70% data as train data form each cluster
train_lat = [i[:TOT_train_points] for i in lat]
train_lon = [i[:TOT_train_points] for i in lon]
train_weekDay = [i[:TOT_train_points] for i in day_of_week]
train_temperature = [i[:TOT_train_points] for i in temperature]
#train_pressure = [i[:TOT_train_points] for i in pressure]
train_weighted_avg = [i[:TOT_train_points] for i in predicted_pickup_values_list]
train_TruePickups = [i[:TOT_train_points] for i in TruePickups]

# Taking 30% data as test data form each cluster
test_lat = [i[TOT_train_points:] for i in lat]
test_lon = [i[TOT_train_points:] for i in lon]
test_weekDay = [i[TOT_train_points:] for i in day_of_week]
test_temperature = [i[TOT_train_points:] for i in temperature]
#test_pressure = [i[TOT_train_points:] for i in pressure]
test_weighted_avg = [i[TOT_train_points:] for i in predicted_pickup_values_list]
test_TruePickups = [i[TOT_train_points:] for i in TruePickups]

# convert from lists of lists of list (tridimensional vector) to lists of list (bidimensional vectors) (VERIFY THE DIMENSIONS OF THOSE DATA STRUCTURES) ----------------------
train_pickups = []
test_pickups = []
train_freq = []
test_freq = []
train_amp = []
test_amp = []
for i in range(NUM_CLUSTERS): # at each iteration put in queue the elements
	train_pickups.extend(train_previousFive_pickups[i]) # here I will have a very long (349*30) vector with each element a 5 element vector
	test_pickups.extend(test_previousFive_pickups[i])
	train_freq.extend(train_fourier_frequencies[i])
	test_freq.extend(test_fourier_frequencies[i])
	train_amp.extend(train_fourier_amplitudes[i])
	test_amp.extend(test_fourier_amplitudes[i])

# Stacking pickups, frequencies and amplitudes (HERE I CUT OFF FREQ AND AMP (seems it is not adding infomrations))
train_prevPickups_freq_amp = np.hstack((train_pickups))
test_prevPickups_freq_amp = np.hstack((test_pickups))

#print("Number of data points in train data = {}. Number of columns till now = {}".format(len(train_prevPickups_freq_amp),
	#len(train_prevPickups_freq_amp[0])))
#print("Number of data points in test data = {}. Number of columns till now = {}".format(len(test_prevPickups_freq_amp),
	#len(test_prevPickups_freq_amp[0])))

# So now we have a two matrices (one for training the other for test) with train_elements*30 and test_elements*30 as rows
# while in the columns we have the 5 pickups, 5 frequencies and 5 amplitudes

# Now we want to convert the lists of lists (matrix) into flat data (single list i.e flatten)

train_flat_lat = sum(train_lat, []) # this function puts all the rows one after the other in one single row
train_flat_lon = sum(train_lon, [])
train_flat_weekDay = sum(train_weekDay, [])
train_flat_temperature = sum(train_temperature, [])
#train_flat_pressure = sum(train_pressure, [])
train_weighted_avg_flat = sum(train_weighted_avg, [])

train_TruePickups_flat = sum(train_TruePickups, []) # TARGET in our training phase. These should be the pickups I want my model to predict

test_flat_lat = sum(test_lat, [])
test_flat_lon = sum(test_lon, [])
test_flat_weekDay = sum(test_weekDay, [])
test_flat_temperature = sum(test_temperature, [])
#test_flat_pressure = sum(test_pressure, [])
test_weighted_avg_flat = sum(test_weighted_avg, [])

test_TruePickups_flat = sum(test_TruePickups, []) # the simulated output will be then compared to that one

print(train_TruePickups_flat[:100])
print(test_TruePickups_flat[:100]) # everytime I run the program, the change, PROBABLY K-MEAN IS INTRODUCING SOME RANDOMENESS
# AND THIS BRINGS TO RANDOMNESS IN THE ORDER OF THE DATA

#------------------------------------------------------------------------------------------------------------------------------------------------------------

# DATAFRAME FOR TENSORFLOW AND XGBOOST (the normalized set is used by the Keras Model)

# TRAIN DATAFRAME
columns = ['ft_5', 'ft_4', 'ft_3', 'ft_2', 'ft_1']
Train_CSV_DF = pd.DataFrame(data = train_pickups, columns = columns) # this is the label column (put as first column)
#Train_CSV_DF = pd.DataFrame(data = train_prevPickups_freq_amp, columns = columns)
Train_CSV_DF["Latitude"] = train_flat_lat
Train_CSV_DF["Longitude"] = train_flat_lon
Train_CSV_DF["WeekDay"] = train_flat_weekDay
Train_CSV_DF["Temperature"] = train_flat_temperature
#Train_CSV_DF["Pressure"] = train_flat_pressure
Train_CSV_DF["Label_Data"] = train_TruePickups_flat
Train_CSV_DF = Train_CSV_DF[['Label_Data', 'ft_5', 'ft_4', 'ft_3', 'ft_2', 'ft_1',
 'Latitude', 'Longitude', 'WeekDay', 'Temperature']]


#TEST DATAFRAME
Test_CSV_DF = pd.DataFrame(data = test_pickups, columns = columns)# this is the label column (put as first column)
#Test_CSV_DF = pd.DataFrame(data = test_prevPickups_freq_amp, columns = columns) 
Test_CSV_DF["Latitude"] = test_flat_lat
Test_CSV_DF["Longitude"] = test_flat_lon
Test_CSV_DF["WeekDay"] = test_flat_weekDay
Test_CSV_DF["Temperature"] = test_flat_temperature
#Test_CSV_DF["Pressure"] = test_flat_pressure
Test_CSV_DF["Label_Data"] = test_TruePickups_flat
Test_CSV_DF = Test_CSV_DF[['Label_Data', 'ft_5', 'ft_4', 'ft_3', 'ft_2', 'ft_1',
 'Latitude', 'Longitude', 'WeekDay', 'Temperature']]

# print some info on the dataframe
print("Shape of train data = "+str(Train_CSV_DF.shape)) # it gives the number of entries and the features of the input. We have 19 features
print("Shape of test data = "+str(Test_CSV_DF.shape))

# The last two columns of the dataset are int32 and tensorflow is complaining about that.
# So I need to cast them as float32
Train_CSV_DF = Train_CSV_DF.astype('float32')
print(Train_CSV_DF.dtypes)
# The last two columns of the dataset are int32 and tensorflow is complaining about that.
# So I need to cast them as float32
Test_CSV_DF = Test_CSV_DF.astype('float32')
print(Test_CSV_DF.dtypes)

# Here I export the dataset for XGBoost (not normalized)
Train_CSV_DF.to_csv(r'C:\Users\Matteo\Desktop\Python_Programs\LaFi_Train_DF_XGBoost_4_0703_NWA_Weather_NOFFT_Hourly.csv', index = False)
Test_CSV_DF.to_csv(r'C:\Users\Matteo\Desktop\Python_Programs\LaFi_Test_DF_XGBoost_4_0703_NWA_Weather_NOFFT_Hourly.csv', index = False)

# Here I am normalizing the columns that we need to normalize (all but the label columns)

cols_to_norm = ['ft_5', 'ft_4', 'ft_3', 'ft_2', 'ft_1', 
				'Latitude', 'Longitude', 'WeekDay', 'Temperature']

Train_CSV_DF[cols_to_norm] = Train_CSV_DF[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
Test_CSV_DF[cols_to_norm] = Test_CSV_DF[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

# Here I export the dataset for Keras (normalized)
Train_CSV_DF.to_csv(r'C:\Users\Matteo\Desktop\Python_Programs\LaFi_Train_DF_Keras_4_0703_NWA_Weather_NOFFT_Hourly.csv', index = False)
Test_CSV_DF.to_csv(r'C:\Users\Matteo\Desktop\Python_Programs\LaFi_Test_DF_Keras_4_0703_NWA_Weather_NOFFT_Hourly.csv', index = False)