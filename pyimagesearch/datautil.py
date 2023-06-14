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

norm_type_list = [None, 'std', 'minmax']
split_mode_list = ["all_year", "month", 'cross_month_validate']
smoothing_mode_list = [None, 'MA', 'EMA']


def get_mode(command, mode_list):
    if type(command) is int:
        if command < 0 or command > len(mode_list):
            return mode_list[0]
        else:
            return mode_list[command]
    else:
        if command in mode_list:
            return command
        else:
            return mode_list[0]


class DataUtil(object):

    def __init__(self, train_path: str,
                 label_col: list = None,
                 feature_col: list = None,
                 normalise=1, label_norm_mode=1,
                 val_path=None, test_path=None,
                 train_split=0.8, val_split=0.1, test_split=0.1,
                 split_mode=False, month_sep=None, keep_date=False,
                 using_images=False, smoothing_mode=None, smoothing_parameter=None, is_val=False):
        # 0.8,0.05,0.15
        """
        使用的天氣欄目
        weather_col
        shift_weather_col 對於使用的天氣欄目，shift時間 例如使用預測日天氣就需要這個
        normalise: norm的方式，default 2 , 0 不做事 ,1 std , 2 minMax
        label_norm_mode: label norm的方式，default 2 , 0 不做事 ,1 std , 2 minMax，優先級比normalise高
        train_split=0.8, val_split=0.05, test_split=0.15
        這三個會100%

        split_mode:
        month_sep:
        keep_date:
        """
        self.using_images = using_images
        self.normalise_mode = get_mode(normalise, norm_type_list)
        self.label_norm_mode = get_mode(label_norm_mode, norm_type_list)
        self.train_df_cloud = None
        self.val_df_cloud = None
        self.test_df_cloud = None
        self.train_df_average = None
        self.val_df_average = None
        self.test_df_average = None
        if self.label_norm_mode == 'std':
            self.labelScaler = StandardScaler()
        elif self.label_norm_mode == 'minmax':
            self.labelScaler = MinMaxScaler()
        else:
            self.labelScaler = None

        # read_dataset
        try:
            self.train_df = pd.read_csv(
                train_path,
                keep_date_col=True,
                parse_dates=["datetime"],
                index_col="datetime")
            assert parameter.data_params.time_granularity is not None
            self.train_df = self.train_df.resample(parameter.data_params.time_granularity).asfreq()
            self.val_df = None
            self.test_df = None
            if (val_path != None):
                self.val_df = pd.read_csv(
                    val_path,
                    parse_dates=["datetime"],
                    index_col="datetime")
                self.val_df = self.val_df.resample(parameter.data_params.time_granularity).asfreq()
            if (test_path != None):
                self.test_df = pd.read_csv(
                    test_path,
                    parse_dates=["datetime"],
                    index_col="datetime")
                self.test_df = self.test_df.resample(parameter.data_params.time_granularity).asfreq()

        except IOError as err:
            print("Error opening data file ... %s", err)
            exit()
            # log.error("Error opening data file ... %s", err)
        ######################資料集切分
        ## split 照比例分
        self.split_mode = get_mode(split_mode, split_mode_list)
        if self.split_mode == "all_year":
            self.train_df, self.val_df = train_test_split(self.train_df, test_size=val_split + test_split,
                                                          shuffle=False)
            self.val_df, self.test_df = train_test_split(self.val_df,
                                                         test_size=test_split / (val_split + test_split),
                                                         shuffle=False)
        elif self.split_mode == "month":
            last_month = np.max(np.unique(self.train_df.index.month))
            assert month_sep is not None
            assert type(month_sep) is int
            assert month_sep <= last_month
            vmonth = month_sep - 2
            train_month = month_sep - 1
            if vmonth <= 0:
                vmonth = last_month + vmonth
            if train_month <= 0:
                train_month = last_month + train_month
            self.test_df = self.train_df[self.train_df.index.month == month_sep]
            self.val_df = self.train_df[self.train_df.index.month == vmonth]
            # self.val_df = self.test_df
            self.train_df = self.train_df[self.train_df.index.month == train_month]
        elif self.split_mode == "cross_month_validate":
            assert month_sep is not None and type(month_sep) is int
            val_month = month_sep - 1 if (month_sep - 1) > 0 else month_sep + 1
            self.test_df = self.train_df[self.train_df.index.month == month_sep]
            self.val_df = self.train_df[self.train_df.index.month == val_month]
            self.train_df = self.train_df[self.train_df.index.month != month_sep]
            self.train_df = self.train_df[self.train_df.index.month != val_month]
        if is_val:
            # leave out test set to avoid biased estimation e.g. when doing hyper-parameters tuning
            self.test_df = self.val_df
        ######################資料急保留時間欄目
        if (keep_date):
            try:
                self.train_df["datetime"] = self.train_df.index
                self.test_df["datetime"] = self.test_df.index
                self.val_df["datetime"] = self.val_df.index
            except:
                log.debug("keep_date some dataset miss")
        # 防呆 如果沒有輸入label_col就全部都作為label
        if label_col is None:
            self.label_col = list(self.train_df.columns)
            self.feature_col = list(self.train_df.columns)
        else:
            if feature_col is None:
                self.label_col = label_col
                self.feature_col = [ele for ele in list(self.train_df.columns) if ele not in label_col]
            else:
                self.label_col = label_col
                self.feature_col = feature_col
        # 如果有需要 會分割time_step轉成需要的欄目
        '''if (parameter.dataUtilParam.time_features):
            self.train_df = self.timeFeatureProcess(self.train_df)
            self.test_df = self.timeFeatureProcess(self.test_df)
            self.val_df = self.timeFeatureProcess(self.val_df)'''
        # smoothing target data
        self.smoothing_mode = get_mode(smoothing_mode, smoothing_mode_list)
        for target in parameter.data_params.target:
            self.train_df[target] = self.smoothing(self.train_df[target], smoothing_mode, smoothing_parameter)
            self.val_df[target] = self.smoothing(self.val_df[target], smoothing_mode, smoothing_parameter)
            self.test_df[target] = self.smoothing(self.test_df[target], smoothing_mode, smoothing_parameter)
            ## 24H to 10H 小時的資料
        if (parameter.data_params.between8_17):
            self.train_df = self.train_df.between_time(parameter.data_params.start, parameter.data_params.end)
            self.val_df = self.val_df.between_time(parameter.data_params.start, parameter.data_params.end)
            self.test_df = self.test_df.between_time(parameter.data_params.start, parameter.data_params.end)

        if parameter.dynamic_model == "two" or ("twoClass" in parameter.inputs):
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
            self.train_df_average[self.train_df_average.isin(["221-240", "201-220", "176-200", "151-175", "0-150"])] = 0
            self.train_df_average[self.train_df_average.isin(["251-255", "241-250"])] = 1
            self.train_df_average = self.train_df_average.astype(np.float32)
            # self.train_df_average = np.expand_dims(self.train_df_average, axis=-1)
            #
            self.val_df_average[self.val_df_average.isin(["221-240", "201-220", "176-200", "151-175", "0-150"])] = 0
            self.val_df_average[self.val_df_average.isin(["251-255", "241-250"])] = 1
            self.val_df_average = self.val_df_average.astype(np.float32)
            # self.val_df_average = np.expand_dims(self.val_df_average, axis=-1)
            #
            self.test_df_average[self.test_df_average.isin(["221-240", "201-220", "176-200", "151-175", "0-150"])] = 0
            self.test_df_average[self.test_df_average.isin(["251-255", "241-250"])] = 1
            self.test_df_average = self.test_df_average.astype(np.float32)
            # self.test_df_average = np.expand_dims(self.test_df_average, axis=-1)
            #
        ##filter data
        self.filter_data()
        # self.column_indices = {name: i for i, name in enumerate(self.train_df.columns)}
        self.normalise_data()
        self.samples_per_day = len(self.train_df.groupby(self.train_df.index.time))
        self.drop_days_with_missing_samples()
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
        self.trainImages = self.load_house_images(self.train_df, parameter.data_params.image_path)
        self.valImages = self.load_house_images(self.val_df, parameter.data_params.image_path)
        self.testImages = self.load_house_images(self.test_df, parameter.data_params.image_path)

    def timeFeatureProcess(self, data):
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['weekday'] = data.index.weekday
        data['hour'] = data.index.hour
        # data['minute'] = dates.date.apply(lambda row: row.minute, 1)
        data['minute'] = data.index.minute

        return data

    def smoothing(self, series, smooth_mode, smooth_parameter):
        if smooth_mode == "MA":
            assert smooth_parameter is type(int)
            smoothed = series.rolling(smooth_parameter).mean()
        elif smooth_mode == "EMA":
            assert smooth_parameter is type(int)
            smoothed = series.ewm(span=smooth_parameter).mean()
        elif smooth_mode is None:
            smoothed = series
        return smoothed

    # 只保留需要的label
    def filter_data(self):
        # log.debug("Normalise: %d", normalise)
        # contain_col = self.label_col + self.weather_col
        contain_col = [ele for ele in list(self.train_df.columns) if (ele in self.feature_col or ele in self.label_col)]
        '''
        if parameter.dataUtilParam.time_features:
            contain_col = contain_col + parameter.dataUtilParam.time_features_col'''
        self.train_df = self.train_df[contain_col].dropna()
        if (self.val_df is not None):
            self.val_df = self.val_df[contain_col].dropna()
        if (self.test_df is not None):
            self.test_df = self.test_df[contain_col].dropna()
        # test if all input data is numerical data, exit if not
        numerical_col = [col_name for col_name, col_type in self.train_df.dtypes.items() if
                         np.issubdtype(col_type, np.number)]
        assert (len(contain_col) == len(numerical_col))

    def normalise_data(self):
        all_data = self.train_df if self.val_df is None else pd.concat([self.train_df, self.val_df], axis=0)
        all_data = all_data if self.test_df is None else pd.concat([all_data, self.test_df], axis=0)
        # normalize label by the scaler specified by user's config
        if self.labelScaler is not None:
            self.labelScaler.fit(all_data[self.label_col])
            self.train_df[self.label_col] = self.labelScaler.transform(self.train_df[self.label_col])
            if (self.val_df is not None):
                self.val_df[self.label_col] = self.labelScaler.transform(self.val_df[self.label_col])
            if (self.test_df is not None):
                self.test_df[self.label_col] = self.labelScaler.transform(self.test_df[self.label_col])
            # log.debug("Normalise: %s", self.norm_type_list[self.normalise_mode])
        # assign different feature scaler corresponding to user input
        if self.normalise_mode == 'std':
            self.scaler = StandardScaler()
        elif self.normalise_mode == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            return
        # feature_scale_col = self.label_col.copy()
        '''if parameter.dataUtilParam.time_features:
            feature_scale_col = feature_scale_col + parameter.dataUtilParam.time_features_col'''
        # exclude label column to avoid repeatedly normalized
        feature_col_excluding_label = [ele for ele in self.feature_col
                                       if ele not in self.label_col or self.labelScaler is None]
        if len(feature_col_excluding_label) > 0:
            # normalize features and fill nan with zero
            self.scaler.fit(all_data[feature_col_excluding_label])
            self.train_df[feature_col_excluding_label] = self.scaler.transform(
                self.train_df[feature_col_excluding_label])
            if (self.val_df is not None):
                self.val_df[feature_col_excluding_label] = self.scaler.transform(
                    self.val_df[feature_col_excluding_label])
            if (self.test_df is not None):
                self.test_df[feature_col_excluding_label] = self.scaler.transform(
                    self.test_df[feature_col_excluding_label])

    def drop_days_with_missing_samples(self):
        # Drop whole-day data if there's any missing observation
        def is_subframe_complete(sub: pd.DataFrame):
            # return True if a subframe(grouped by day) has no missing sample
            return len(sub.index.time) == self.samples_per_day

        self.train_df = self.train_df.groupby(self.train_df.index.date).filter(is_subframe_complete)
        self.val_df = self.val_df.groupby(self.val_df.index.date).filter(is_subframe_complete)
        self.test_df = self.test_df.groupby(self.test_df.index.date).filter(is_subframe_complete)

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

    def mask(self, image, x0=32, y0=22, r=26):  # (image, x0=32, y0=22, r=26):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, (x0, y0), r, 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)
        if (masked.ndim < 3):
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
        if not self.using_images:
            return None
        # initialize our images array (i.e., the house images themselves)
        images = []

        df.index = df.index.tz_localize(parameter.data_params.timezone)
        # loop over the indexes of the houses
        for i in df.index:
            # print(str(df["datetime"][i])[5:7])
            m = str(i)[5:7] + str(i)[8:10]
            n = str(i)[11:13] + str(i)[14:16]
            basePath = os.path.sep.join(["../skyImage/2020/2020{}/".format(m) + "NCU_skyimg_2020{}_{}*".format(m, n)])
            housePaths = sorted(list(glob.glob(basePath)))
            for housePath in housePaths:
                # load the input image, resize it to be 32 32, and then
                # update the list of input images
                # print(housePath)
                image = cv2.imread(housePath)
                image = cv2.resize(image, (64, 48))
                # cv2.imshow('img', image)
                image = self.mask(image)
                if parameter.data_params.is_using_sun_location:
                    sun_region = self.locate_sun(i)
                    sun_region = self.mask(sun_region)
                    image = np.concatenate((image, sun_region), axis=-1)
                if parameter.data_params.is_using_cloud_location:
                    cloud_region = self.binarize_by_BR_ratio(image)
                    cloud_region = self.mask(cloud_region)
                    image = np.concatenate((image, cloud_region), axis=-1)
                images.append(image)

                # cv2.waitKey(0)
                ##########################################################################################
        # return our set of images
        df.index = df.index.tz_localize(None)
        return np.array(images) / 255.0
