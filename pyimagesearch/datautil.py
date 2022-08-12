import numpy as np

# Logging
import logging
# from __main__ import logger_name
# log = logging.getLogger(logger_name)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyimagesearch import parameter
from pyimagesearch import Msglog
from pvlib import solarposition
import math
import os
import glob
import cv2

class DataUtil(object):

    def __init__(self, train_path: str,
                 label_col: list = None,
                 feature_col: list = None,
                 normalise=2,
                 val_path=None, test_path=None,
                 train_split=0.8, val_split=0.05, test_split=0.15, split_mode=False, month_sep=None, keep_date=False):
        #0.8,0.05,0.15
        """
        使用的天氣欄目
        weather_col
        shift_weather_col 對於使用的天氣欄目，shift時間 例如使用預測日天氣就需要這個
        normalise: norm的方式，default 2 , 0 不做事 ,1 std , 2 minMax

        train_split=0.8, val_split=0.05, test_split=0.15
        這三個會100%

        split_mode:
        month_sep:
        keep_date:
        """
        self.norm_type_list = ['None', 'Stander', 'MinMax']
        self.normalise_mode = normalise
        self.train_df_cloud = None
        self.val_df_cloud = None
        self.test_df_cloud = None
        self.train_df_average = None
        self.val_df_average = None
        self.test_df_average = None
        # read_dataset
        try:
            self.train_df = pd.read_csv(
                train_path,
                keep_date_col=True,
                parse_dates=["datetime"],
                index_col="datetime")
            self.val_df = None
            self.test_df = None
            if (val_path != None):
                self.val_df = pd.read_csv(
                    val_path,
                    parse_dates=["datetime"],
                    index_col="datetime")

            if (test_path != None):
                self.test_df = pd.read_csv(
                    test_path,
                    parse_dates=["datetime"],
                    index_col="datetime")

        except IOError as err:
            print("Error opening data file ... %s", err)
            # log.error("Error opening data file ... %s", err)
        ######################資料集切分
        ## split 照比例分
        if (split_mode == "all_year"):
            if parameter.input_days is None or parameter.output_days is None:
                self.train_df, self.val_df = train_test_split(self.train_df, test_size=val_split + test_split,
                                                              shuffle=False)
                self.val_df, self.test_df = train_test_split(self.val_df, test_size=test_split / (val_split + test_split),
                                                            shuffle=False)
            else:
                assert month_sep is not None and type(month_sep) is int
                self.test_df = self.train_df[self.train_df.index.month == month_sep]
                self.train_df = self.train_df[self.train_df.index.month != month_sep]
                all_dates = np.unique(self.train_df.index.date)
                train_dates = all_dates[:int((1-val_split)*len(all_dates))]
                val_dates = all_dates[int((1 - val_split) * len(all_dates)):]
                self.val_df = self.train_df[[True if (_ in val_dates) else False for _ in self.train_df.index.date]]
                self.train_df = self.train_df[[True if (_ in train_dates) else False for _ in self.train_df.index.date]]
        elif (split_mode == "month"):
            assert month_sep is not None and type(month_sep) is int
            vmonth = month_sep - 2
            train_month = month_sep - 1
            if vmonth <= 0:
                vmonth = parameter.tailMonth + vmonth
            if train_month <= 0:
                train_month = parameter.tailMonth + train_month
            self.test_df = self.train_df[self.train_df.index.month == month_sep]
            self.val_df = self.train_df[self.train_df.index.month == vmonth]
            # self.val_df = self.test_df
            self.train_df = self.train_df[self.train_df.index.month == train_month]
        ######################資料急保留時間欄目
        if (keep_date):
            try:
                self.train_df["datetime"] = self.train_df.index
                self.test_df["datetime"] = self.test_df.index
                self.val_df["datetime"] = self.val_df.index
            except:
                log.debug("keep_date some dataset miss")
        #防呆 如果沒有輸入label_col就全部都作為label
        if label_col is None:
            self.label_col = list(self.train_df.columns)
            self.feature_col = list(self.train_df.columns)
        else:
            if feature_col is None:
                self.label_col = label_col
                self.feature_col = list(set(list(self.train_df.columns)) - set(label_col))
            else:
                self.label_col = label_col
                self.feature_col = feature_col
        # 如果有需要 會分割time_step轉成需要的欄目
        '''if (parameter.dataUtilParam.time_features):
            self.train_df = self.timeFeatureProcess(self.train_df)
            self.test_df = self.timeFeatureProcess(self.test_df)
            self.val_df = self.timeFeatureProcess(self.val_df)'''
        # smoothing target data
        if parameter.smoothing_type in parameter.smoothing_mode:
            for target in parameter.target:
                self.train_df[target] = self.smoothing(self.train_df[target])
                self.val_df[target] = self.smoothing(self.val_df[target])
                self.test_df[target] = self.smoothing(self.test_df[target])
            ## 24H to 10H 小時的資料
        if (parameter.between8_17):
            self.train_df = self.train_df.between_time('08:00:01', '17:00:00')
            self.val_df = self.val_df.between_time('08:00:01', '17:00:00')
            self.test_df = self.test_df.between_time('08:00:01', '17:00:00')


        if parameter.dynamic_model=="two" or ("twoClass" in parameter.inputs):
            self.train_df_cloud = self.train_df["twoClass"].astype(np.str)
            self.val_df_cloud = self.val_df["twoClass"].astype(np.str)
            self.test_df_cloud = self.test_df["twoClass"].astype(np.str)
            self.train_df_cloud[self.train_df_cloud.isin(['a'])] = 1
            self.train_df_cloud[self.train_df_cloud.isin(['c'])] = 0
            self.train_df_cloud = self.train_df_cloud.astype(np.float32)
            # self.train_df_cloud = np.expand_dims(self.train_df_cloud, axis=-1)
            #
            self.val_df_cloud[self.val_df_cloud.isin(['a'])] = 1
            self.val_df_cloud[self.val_df_cloud.isin(['c'])] = 0
            self.val_df_cloud = self.val_df_cloud.astype(np.float32)
            # self.val_df_cloud = np.expand_dims(self.val_df_cloud, axis=-1)
            #
            self.test_df_cloud[self.test_df_cloud.isin(['a'])] = 1
            self.test_df_cloud[self.test_df_cloud.isin(['c'])] = 0
            self.test_df_cloud = self.test_df_cloud.astype(np.float32)
            # self.test_df_cloud = np.expand_dims(self.test_df_cloud, axis=-1)
            #
        if ("sun_average" in parameter.inputs):
            self.train_df_average = self.train_df["sun_average"].astype(np.str)
            self.val_df_average = self.val_df["sun_average"].astype(np.str)
            self.test_df_average = self.test_df["sun_average"].astype(np.str)
            self.train_df_average[self.train_df_average.isin(["221-240","201-220","176-200","151-175","0-150"])] = 0
            self.train_df_average[self.train_df_average.isin(["251-255","241-250"])] = 1
            self.train_df_average = self.train_df_average.astype(np.float32)
            # self.train_df_average = np.expand_dims(self.train_df_average, axis=-1)
            #
            self.val_df_average[self.val_df_average.isin(["221-240","201-220","176-200","151-175","0-150"])] = 0
            self.val_df_average[self.val_df_average.isin(["251-255","241-250"])] = 1
            self.val_df_average = self.val_df_average.astype(np.float32)
            # self.val_df_average = np.expand_dims(self.val_df_average, axis=-1)
            #
            self.test_df_average[self.test_df_average.isin(["221-240","201-220","176-200","151-175","0-150"])] = 0
            self.test_df_average[self.test_df_average.isin(["251-255","241-250"])] = 1
            self.test_df_average = self.test_df_average.astype(np.float32)
            # self.test_df_average = np.expand_dims(self.test_df_average, axis=-1)
            #
        ##filter data
        # self.filter_data()
        # self.column_indices = {name: i for i, name in enumerate(self.train_df.columns)}
        self.normalise_data()
        print("data Preprocess")
        print(self.train_df)
        print(self.val_df)
        print(self.test_df)
        # self.train_df = self.train_df.join(self.train_df_cloud)
        # self.val_df = self.val_df.join(self.val_df_cloud)
        # self.test_df = self.test_df.join(self.test_df_cloud)
        # print(self.train_df)
        # print(self.val_df)
        # print(self.test_df)
        self.trainImages = self.load_house_images(self.train_df, parameter.datasetPath)
        self.valImages = self.load_house_images(self.val_df, parameter.datasetPath)
        self.testImages = self.load_house_images(self.test_df, parameter.datasetPath)



    def timeFeatureProcess(self, data):
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['weekday'] = data.index.weekday
        data['hour'] = data.index.hour
        # data['minute'] = dates.date.apply(lambda row: row.minute, 1)
        data['minute'] = data.index.minute

        return data

    def smoothing(self, series):
        if parameter.smoothing_type == "MA":
            smoothed = series.rolling(parameter.smoothing_mode["MA"]["num_of_entries"]).mean()
        elif parameter.smoothing_type == "EMA":
            smoothed = series.ewm(span = parameter.smoothing_mode["EMA"]["span"]).mean()
        else:
            smoothed = series
        return smoothed
    
    # 只保留需要的label
    def filter_data(self):
        # log.debug("Normalise: %d", normalise)
        # contain_col = self.label_col + self.weather_col
        contain_col = self.label_col

        '''
        if parameter.dataUtilParam.time_features:
            contain_col = contain_col + parameter.dataUtilParam.time_features_col'''
        self.train_df = self.train_df[contain_col]
        if (self.val_df is not None):
            self.val_df = self.val_df[contain_col]
        if (self.test_df is not None):
            self.test_df = self.test_df[contain_col]
            
    # norm
    def normalise_data(self):
        # log.debug("Normalise: %s", self.norm_type_list[self.normalise_mode])
        if self.normalise_mode == 0:  # do not normalise
            self.scaler = None
            self.labelScaler = None
            return
        if self.normalise_mode == 1:  # stand scaler
            self.scaler = StandardScaler()
            self.labelScaler = StandardScaler()
        if self.normalise_mode == 2:  # minmax scaler
            self.scaler = MinMaxScaler()
            self.labelScaler = MinMaxScaler()
        # feature_scale_col = self.label_col.copy()
        '''if parameter.dataUtilParam.time_features:
            feature_scale_col = feature_scale_col + parameter.dataUtilParam.time_features_col'''
        self.labelScaler.fit(self.train_df[self.label_col])
        self.train_df[self.label_col] = self.labelScaler.transform(self.train_df[self.label_col])
        if (self.val_df is not None):
            self.val_df[self.label_col] = self.labelScaler.transform(self.val_df[self.label_col])
        if (self.test_df is not None):
            self.test_df[self.label_col] = self.labelScaler.transform(self.test_df[self.label_col])
        # exclude label column to avoid repeatedly normalized
        feature_col_excluding_label = list(set(self.feature_col) - set(self.label_col))
        if len(feature_col_excluding_label) > 0:
            self.scaler.fit(self.train_df[feature_col_excluding_label])
            self.train_df[feature_col_excluding_label] = self.scaler.transform(
                self.train_df[feature_col_excluding_label])
            if (self.val_df is not None):
                self.val_df[feature_col_excluding_label] = self.scaler.transform(self.val_df[feature_col_excluding_label])
            if (self.test_df is not None):
                self.test_df[feature_col_excluding_label] = self.scaler.transform(self.test_df[self.feature_col_excluding_label])

    def __repr__(self):
        return '\n'.join([
            f'##################################################################################################',
            f'TrainSet shape: {self.train_df.shape}',
            f'ValSet shape: {self.val_df.shape}',
            f'TestSet shape: {self.test_df.shape}',
            f'Label : {self.label_col}',
            f'Norm mode: {self.normalise_mode}',
            f'##################################################################################################'
        ])

    def mask(self, image, x0=32, y0=22, r=26): #(image, x0=32, y0=22, r=26):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, (x0, y0), r, 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)
        if(masked.ndim < 3):
            masked = np.expand_dims(masked, axis=-1)
        return masked

    def locate_sun(self, date_time, target_dim=(48, 64), center=(32, 22), view_radius=26, sun_radius=5):
        latitude, longitude = 24.969367, 121.190733
        spa = solarposition.spa_python(date_time, latitude, longitude)

        azimuth = spa["azimuth"][date_time]
        zenith = spa["zenith"][date_time]
        elevation = spa["elevation"][date_time]
        PI = 3.1415926535897932384626433832795028841971
        x = -(zenith * math.cos((270 - azimuth) * PI / 180) / 90) * view_radius + center[0]
        y = -(zenith * math.sin((270 - azimuth) * PI / 180) / 90) * view_radius + center[1]

        sun_region = np.zeros(target_dim, dtype="uint8")
        cv2.circle(sun_region, (int(x), int(y)), sun_radius, 255, -1)
        # cv2.imshow('sun', sun_region)
        sun_region = np.expand_dims(sun_region, axis=-1)
        return sun_region

    def binarize_by_BR_ratio(self, image, thershold=0.04):
        # based on https://www.researchgate.net/publication/258495829_A_Hybrid_Thresholding_Algorithm_for_Cloud_Detection_on_Ground-Based_Color_Images
        B = image[:, :, 0].astype('float64')
        R = image[:, :, -1].astype('float64')
        normalized_BR_ratio_map = (B - R) / (B + R + 1e-5)
        _, binarized = cv2.threshold(normalized_BR_ratio_map, thershold, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow('cloud', binarized)
        binarized = np.expand_dims(binarized.astype('uint8'), axis=-1)
        return binarized

    def load_house_images(self, df, inputPath):
        # initialize our images array (i.e., the house images themselves)
        images = []

        df.index = df.index.tz_localize(parameter.timezone)
        # loop over the indexes of the houses
        for i in df.index:
            if not parameter.is_using_image_data:
                break
            # print(str(df["datetime"][i])[5:7])
            m = str(i)[5:7] + str(i)[8:10]
            n = str(i)[11:13] + str(i)[14:16]
            basePath = os.path.sep.join(["../skyImage/2020/2020{}/".format(m)+"NCU_skyimg_2020{}_{}*".format(m,n)])
            housePaths = sorted(list(glob.glob(basePath)))
            for housePath in housePaths:
                # load the input image, resize it to be 32 32, and then
                # update the list of input images
                # print(housePath)
                image = cv2.imread(housePath)
                image = cv2.resize(image, (64, 48))
                # cv2.imshow('img', image)
                image = self.mask(image)
                if parameter.is_using_sun_location:
                    sun_region = self.locate_sun(i)
                    sun_region = self.mask(sun_region)
                    image = np.concatenate((image, sun_region), axis=-1)
                if parameter.is_using_cloud_location:
                    cloud_region = self.binarize_by_BR_ratio(image)
                    cloud_region = self.mask(cloud_region)
                    image = np.concatenate((image, cloud_region), axis=-1)
                images.append(image)

                # cv2.waitKey(0)
                ##########################################################################################
        # return our set of images
        df.index = df.index.tz_localize(None)
        return np.array(images) / 255.0