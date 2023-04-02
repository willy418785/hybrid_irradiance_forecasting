# import the necessary packages
import pandas as pd
import numpy as np
import glob
import cv2
import os
import re
from sklearn.utils import shuffle
from pyimagesearch import parameter
from pvlib import solarposition
from datetime import datetime
import math

def load_house_attributes(inputPath, mode=1, month_sep=None):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	# cols = ["datetime", "ShortWaveDown", "twoClass"]
	# df = pd.read_csv(inputPath, sep=",", header=None, names=cols)
	df = pd.read_csv(inputPath, keep_date_col=True, parse_dates=["datetime"], index_col="datetime")
	df["datetime"] = df.index
	df = df.between_time('08:00:01', '17:00:00')
	# if parameter.static_suffle == True:
	# 	# print(df)
	# 	df = shuffle(df)
	# 	print(df)
	if parameter.data_params.split_mode =="month":
		vmonth = month_sep - 2
		if vmonth == 0:
			vmonth = parameter.tailMonth
		test_df = df[df.index.month == month_sep]
		val_df = df[df.index.month == vmonth]
		train_df = df[df.index.month == month_sep - 1]
		return train_df, val_df, test_df
	elif mode==1:
		if parameter.static_suffle == True:
			# print(df)
			df = shuffle(df)
			# print(df)
		return df
	elif mode==2:
		groups = df.groupby(df["sun_average"])    #twoClass
		df1_1 = groups.get_group("0-150")
		df1_2 = groups.get_group("151-175")
		df1_3 = groups.get_group("176-200")
		# df1_4 = groups.get_group("201-220")
		# df1_5 = groups.get_group("221-240")
		df1 = pd.concat( [df1_1, df1_2, df1_3], axis=0 )

		df2_1 = groups.get_group("251-255")  
		df2_2 = groups.get_group("241-250") 
		df2_3 = groups.get_group("221-240") 
		df2_4 = groups.get_group("201-220")  

		df2 = pd.concat( [df2_1, df2_2, df2_3, df2_4], axis=0 ) 

		print(df1, df2)
		if parameter.static_suffle == True:
			# print(df)
			df1 = shuffle(df1)
			df2 = shuffle(df2)
			print(df1, df2)
		return df1, df2
	elif mode==3:
		groups = df.groupby(df["twoClass"])    #twoClass
		df1 = groups.get_group("a")
		df2 = groups.get_group("c") 
		print(df1, df2)
		if parameter.static_suffle == True:
			# print(df)
			df1 = shuffle(df1)
			df2 = shuffle(df2)
			print(df1, df2)
		return df1, df2
	elif mode==4:
		groups = df.groupby(df["sun_kmean4"])
		df1 = groups.get_group("dark")
		df2 = groups.get_group("medium")  
		df3 = groups.get_group("bright")

		if parameter.static_suffle == True:
			# print(df)
			df1 = shuffle(df1)
			df2 = shuffle(df2)
			df3 = shuffle(df3)
		return df1, df2, df3

def mask(image, x0=32, y0=22, r=26): #(image, x0=32, y0=22, r=26):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (x0, y0), r, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked
# def mask(image, x0=64, y0=44, r=52): #(image, x0=32, y0=22, r=26):
#     mask = np.zeros(image.shape[:2], dtype="uint8")
#     cv2.circle(mask, (x0, y0), r, 255, -1)
#     masked = cv2.bitwise_and(image, image, mask=mask)
#     return masked

def load_house_images(df, inputPath):
	# initialize our images array (i.e., the house images themselves)
	images = []

	# loop over the indexes of the houses
	for i in df.index.values:
		# print(str(df["datetime"][i])[5:7])
		m = str(df["datetime"][i])[5:7] + str(df["datetime"][i])[8:10]
		n = str(df["datetime"][i])[11:13] + str(df["datetime"][i])[14:16]
		# print(m,n)
		# print(n)

		basePath = os.path.sep.join(["../skyImage/2020/2020{}/".format(m)+"NCU_skyimg_2020{}_{}*".format(m,n)])
		housePaths = sorted(list(glob.glob(basePath)))
		for housePath in housePaths:
			# load the input image, resize it to be 32 32, and then
			# update the list of input images
			# print(housePath)
			image = cv2.imread(housePath)
			# image = cv2.imread(housePath,cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image, (64, 48))
			# image = cv2.resize(image, (128, 96))
			# print("############",image.shape)
			image = mask(image)
			# image = Kmeans(image)
			# cv2.imwrite(str(m+n)+".jpg",image)
			images.append(image)   
			##########################################################################################
	# return our set of images
	return np.array(images) 

def data_preprocess(df=None, train_df=None, val_df=None, test_df=None):
	if parameter.data_params.split_mode == "all_year":
		images = load_house_images(df, parameter.data_params.image_path)
		images = images / 255.0
		num_val_samples = len(images) // 10
		trainAttrX, valAttrX, testAttrX = df[0 : 8*num_val_samples], df[8*num_val_samples : 9*num_val_samples], df[9*num_val_samples : 10*num_val_samples]
		trainImagesX, valImageX, testImagesX = images[0 : 8*num_val_samples], images[8*num_val_samples : 9*num_val_samples], images[9*num_val_samples : 10*num_val_samples]
	elif parameter.data_params.split_mode == "month":
		trainAttrX, valAttrX, testAttrX = train_df, val_df, test_df
		trainImagesX = load_house_images(trainAttrX, parameter.data_params.image_path)
		trainImagesX = trainImagesX / 255.0
		valImageX = load_house_images(valAttrX, parameter.data_params.image_path)
		valImageX = valImageX / 255.0
		testImagesX = load_house_images(testAttrX, parameter.data_params.image_path)
		testImagesX = testImagesX / 255.0

	# maxPrice = trainAttrX["ShortWaveDown"].max()
	# trainY_nor = trainAttrX["ShortWaveDown"] / maxPrice
	# valY_nor = valAttrX["ShortWaveDown"] / maxPrice
	# testY = testAttrX["ShortWaveDown"]
	# testIndex = testAttrX["datetime"]
	maxPrice = trainAttrX[parameter.data_params.target].max()
	minPrice = trainAttrX[parameter.data_params.target].min()
	meanPrice = trainAttrX[parameter.data_params.target].mean()
	stdPrice = trainAttrX[parameter.data_params.target].std()
	
	trainCloudX = trainAttrX[parameter.cloudLabel].astype(np.str)
	trainCloudX[trainCloudX.isin(['a'])] = 0
	trainCloudX[trainCloudX.isin(['c'])] = 1
	trainCloudX = trainCloudX.astype(np.float32)
	
	valCloudX = valAttrX[parameter.cloudLabel].astype(np.str)
	valCloudX[valCloudX.isin(['a'])] = 0
	valCloudX[valCloudX.isin(['c'])] = 1
	valCloudX = valCloudX.astype(np.float32)

	testCloudX = testAttrX[parameter.cloudLabel].astype(np.str)
	testCloudX[testCloudX.isin(['a'])] = 0
	testCloudX[testCloudX.isin(['c'])] = 1
	testCloudX = testCloudX.astype(np.float32)
	
	trainCloudX = np.expand_dims(trainCloudX, axis=-1)
	valCloudX = np.expand_dims(valCloudX, axis=-1)
	testCloudX = np.expand_dims(testCloudX, axis=-1)
	
	if parameter.normalization == "MinMax":
		trainDataX = (trainAttrX[parameter.data_params.target]-minPrice) / (maxPrice-minPrice)
		valDataX = (valAttrX[parameter.data_params.target]-minPrice) / (maxPrice-minPrice)
		testDataX = (testAttrX[parameter.data_params.target]-minPrice) / (maxPrice-minPrice)
	elif parameter.normalization == "Mean":
		trainDataX = (trainAttrX[parameter.data_params.target]-meanPrice) / (maxPrice-minPrice)
		valDataX = (valAttrX[parameter.data_params.target]-meanPrice) / (maxPrice-minPrice)
		testDataX = (testAttrX[parameter.data_params.target]-meanPrice) / (maxPrice-minPrice)
	elif parameter.normalization == "Standard":
		trainDataX = (trainAttrX[parameter.data_params.target]-meanPrice) / stdPrice
		valDataX = (valAttrX[parameter.data_params.target]-meanPrice) / stdPrice
		testDataX = (testAttrX[parameter.data_params.target]-meanPrice) / stdPrice
	elif parameter.normalization == "Max":
		trainDataX = trainAttrX[parameter.data_params.target] / maxPrice
		valDataX = valAttrX[parameter.data_params.target] / maxPrice
		testDataX = testAttrX[parameter.data_params.target] / maxPrice
	else:
		trainDataX = trainAttrX[parameter.data_params.target]
		valDataX = valAttrX[parameter.data_params.target]
		testDataX = testAttrX[parameter.data_params.target]
	
	trainDataX = np.expand_dims(trainDataX, axis=-1)
	valDataX = np.expand_dims(valDataX, axis=-1)
	testDataX = np.expand_dims(testDataX, axis=-1)

	trainY_nor, valY_nor= trainDataX, valDataX
	testY = testAttrX["ShortWaveDown"]
	testY = np.expand_dims(testY, axis=-1)
	testIndex = testAttrX["datetime"]
	testIndex = np.expand_dims(testIndex, axis=-1)

	test_before = testAttrX[parameter.targetAdd]
	test_before = np.expand_dims(test_before, axis=-1)

	return test_before, trainDataX, trainCloudX, valDataX, valCloudX, testDataX, testCloudX, trainImagesX, valImageX, testImagesX, trainY_nor, valY_nor, testY, maxPrice, minPrice, meanPrice, stdPrice, testIndex

def data_preprocess_static(images, df):
	num_val_samples = len(images) // 10
	trainAttrX, valAttrX, testAttrX = df[0 : 8*num_val_samples], df[8*num_val_samples : 9*num_val_samples], df[9*num_val_samples : 10*num_val_samples]
	trainImagesX, valImageX, testImagesX = images[0 : 8*num_val_samples], images[8*num_val_samples : 9*num_val_samples], images[9*num_val_samples : 10*num_val_samples]
	maxPrice = trainAttrX["ShortWaveDown"].max()
	trainY_nor = trainAttrX["ShortWaveDown"] / maxPrice
	valY_nor = valAttrX["ShortWaveDown"] / maxPrice
	testY = testAttrX["ShortWaveDown"]
	testIndex = testAttrX["datetime"]
	trainY_nor = np.expand_dims(trainY_nor, axis=-1)
	valY_nor = np.expand_dims(valY_nor, axis=-1)
	testY = np.expand_dims(testY, axis=-1)
	testIndex = np.expand_dims(testIndex, axis=-1)
	return trainImagesX, valImageX, testImagesX, trainY_nor, valY_nor, testY, maxPrice, testIndex