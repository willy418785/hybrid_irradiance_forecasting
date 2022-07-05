import pandas as pd
from pathlib import Path
import numpy as np
from pvlib import solarposition
from datetime import datetime
import math
import os
import glob
import cv2

def mask(image, x0=265, y0=186, r=180): #(image, x0=32, y0=22, r=26):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (x0, y0), r, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

def Kmeans(image):
	# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	twoDimage = image.reshape((-1,1))
	twoDimage = np.float32(twoDimage)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 4
	attempts=10
	ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	result_image = res.reshape((image.shape))
	# cv2.circle(result_image,(int(x),int(y)),30,(0,0,255),3)
	return result_image

def load_house_images(df, inputPath):
    df["datetime"] = df.index
    df = df.between_time('08:00:01', '17:00:00')
    df2 = pd.DataFrame()
    sun_list = []
    time_list = []
    # initialize our images array (i.e., the house images themselves)
    images = []

    tz = 'Asia/Taipei'
    latitude, longitude = 24.969367, 121.190733       #24°58'6"N   121°11'27"E    #121.190733,24.969367  #NCU
    time = pd.date_range('2020-01-01 08:00:00', '2020-08-31 17:01:00', closed='left', freq='min', tz=tz)
    spa = solarposition.spa_python(time, latitude, longitude)
    print(spa)
    
    # loop over the indexes of the houses
    for i in df.index.values:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        # print(str(df["datetime"][i])[5:7])
        # print(str(df["datetime"][i]))
        m = str(df["datetime"][i])[5:7] + str(df["datetime"][i])[8:10]
        n = str(df["datetime"][i])[11:13] + str(df["datetime"][i])[14:16]
        # print(m,n)
        # print(n)
        time = datetime(2020, int(m[0:2]), int(m[2:4]), int(n[0:2]), int(n[2:4]))
        azimuth = spa["azimuth"][time]
        zenith = spa["zenith"][time]
        elevation = spa["elevation"][time]
        PI = 3.1415926535897932384626433832795028841971
        x = -(zenith * math.cos((270-azimuth)*PI/180) / 90) * 200 + 265
        y = -(zenith * math.sin((270-azimuth)*PI/180) / 90) * 200 + 186
        
        basePath = os.path.sep.join(["../skyImage/2020/2020{}/".format(m)+"NCU_skyimg_2020{}_{}*".format(m,n)])
        housePaths = sorted(list(glob.glob(basePath)))
        for housePath in housePaths:
            # load the input image, resize it to be 32 32, and then
            # update the list of input images
            # print(housePath)
            image = cv2.imread(housePath,cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (512, 384))
            # print("############",image.shape)
            image = mask(image)
            image = Kmeans(image)
            # cv2.imwrite(str(m+n)+".jpg",image)
            images.append(image)   
            ##########################################################################################

            dir_p = {}
            r = 30
            # print(result_image[(int(y)-r):(int(y)+r),(int(x)-r):(int(x)+r)])
            for i in range((int(y)-r),(int(y)+r)):
                for j in range((int(x)-r),(int(x)+r)):
                    if image[i,j] not in dir_p:
                        # print(image[i,j])
                        dir_p[image[i,j]] = 0
                    dir_p[image[i,j]] += 1
            num = 0
            flag = 0
            '''for key in sorted(dir_p):
                print(key, dir_p[key])
            print(len(dir_p))'''
            
            for key in sorted(dir_p):
                if key!=0 and key!=1:
                    if len(dir_p)==1:
                        if key<190:
                            flag += 2
                        else:
                            flag += 1

                    elif len(dir_p)==2:
                        if dir_p[key]>2400:#300
                            if key<190:
                                flag += 2
                            else:
                                flag += 1
                    elif len(dir_p)==3:
                        if dir_p[key]>1800:#200
                            if key<190:
                                flag += 2
                            else:
                                flag += 1
                    elif len(dir_p)==4:
                        if dir_p[key]>900:#150
                            if key<190:
                                flag += 2
                            else:
                                flag += 1
            # if flag == 2 or flag == 4:
            #     print(housePath," dark")
            # elif flag == 1:
            #     print(housePath," bright")
            # elif flag == 0 or flag == 3 or flag == 5:
            #     print(housePath," medium")
            if flag == 2 or flag == 4:
                # print(housePath," dark")
                time_list.append(time)
                sun_list.append("dark")
            elif flag == 1:
                # print(housePath," bright")
                time_list.append(time)
                sun_list.append("bright")
            elif flag == 0 or flag == 3 or flag == 5:
                # print(housePath," medium")
                time_list.append(time)
                sun_list.append("medium")
    df2 = pd.DataFrame({'datetime' : time_list, 'sun_kmean4' : sun_list}) 
    print(df2)
            ##########################################################################################
    return df2
    

def average(df, inputPath):
    df["datetime"] = df.index
    df = df.between_time('08:00:01', '17:00:00')
    df2 = pd.DataFrame()
    sun_list = []
    time_list = []
    # initialize our images array (i.e., the house images themselves)
    images = []
    tz = 'Asia/Taipei'
    latitude, longitude = 24.969367, 121.190733       #24°58'6"N   121°11'27"E    #121.190733,24.969367  #NCU
    time = pd.date_range('2020-01-01 08:00:00', '2020-08-31 17:01:00', closed='left', freq='min', tz=tz)
    spa = solarposition.spa_python(time, latitude, longitude)
    # print(spa)

    # loop over the indexes of the houses
    for i in df.index.values:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        # print(str(df["datetime"][i])[5:7])
        # print(str(df["datetime"][i]))
        m = str(df["datetime"][i])[5:7] + str(df["datetime"][i])[8:10]
        n = str(df["datetime"][i])[11:13] + str(df["datetime"][i])[14:16]
        # print(m,n)
        # print(n)
        time = datetime(2020, int(m[0:2]), int(m[2:4]), int(n[0:2]), int(n[2:4]))
        azimuth = spa["azimuth"][time]
        zenith = spa["zenith"][time]
        elevation = spa["elevation"][time]
        PI = 3.1415926535897932384626433832795028841971
        x = -(zenith * math.cos((270-azimuth)*PI/180) / 90) * 200 + 265
        y = -(zenith * math.sin((270-azimuth)*PI/180) / 90) * 200 + 186

        basePath = os.path.sep.join(["../skyImage/2020/2020{}/".format(m)+"NCU_skyimg_2020{}_{}*".format(m,n)])
        housePaths = sorted(list(glob.glob(basePath)))
        for housePath in housePaths:
            # load the input image, resize it to be 32 32, and then
            # update the list of input images
            # print(housePath)
            image = cv2.imread(housePath,cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (512, 384))
            # print("############",image.shape)
            image = mask(image)
            # cv2.imwrite(str(m+n)+".jpg",image)
            # images.append(image)   
            ##########################################################################################

            dir_p = {}
            r = 30
            sumPixel = 0
            sumNum = 0
            # print(result_image[(int(y)-r):(int(y)+r),(int(x)-r):(int(x)+r)])
            for i in range((int(y)-r),(int(y)+r)):
                for j in range((int(x)-r),(int(x)+r)):
                    if image[i,j] != 0:
                        sumNum += 1
                        sumPixel += image[i,j]
                    #     dir_p[image[i,j]] = 0
                    # dir_p[image[i,j]] += 1
            # print(housePath,sumPixel/sumNum)
            if (sumPixel/sumNum) > 250:
                time_list.append(time)
                sun_list.append("251-255")
            elif (sumPixel/sumNum) > 240:
                time_list.append(time)
                sun_list.append("241-250")
            elif (sumPixel/sumNum) > 220:
                time_list.append(time)
                sun_list.append("221-240")
            elif (sumPixel/sumNum) > 200:
                time_list.append(time)
                sun_list.append("201-220")
            elif (sumPixel/sumNum) > 175:
                time_list.append(time)
                sun_list.append("176-200")
            elif (sumPixel/sumNum) > 150:
                time_list.append(time)
                sun_list.append("151-175")
            else:
                time_list.append(time)
                sun_list.append("0-150")
    df2 = pd.DataFrame({'datetime' : time_list, 'sun_average' : sun_list}) 
    # print(df2)
    return df2

def generateEntireDataset(inputPath):
    df_data = pd.read_csv(inputPath, keep_date_col=True, parse_dates=['datetime'], index_col="datetime")
    # df_data = df_data.sort_index()
    # (option) concat history weather 加入歷史氣象
    
    print(df_data)
    dataset = "../skyImage"
    df2 = load_house_images(df_data, dataset)
    df3 = average(df_data, dataset)

    df_data.index.name = None
    df_final = pd.merge(df_data, df2, how='left', on=['datetime'])
    df_final = pd.merge(df_final, df3, how='left', on=['datetime'])
    df_final.set_index('datetime',inplace = True)
    print(df_final)
    output_path = Path('2020new.csv')

    df_final.to_csv(output_path, encoding='utf8')


if __name__ == '__main__':
    generateEntireDataset("2020final.csv")
