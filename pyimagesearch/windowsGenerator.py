import tensorflow as tf
import numpy as np
from pyimagesearch import parameter
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import plotly.graph_objects as go
from pathlib import Path
from pyimagesearch import my_metrics
import os
import gc


class WindowGenerator():
    def __init__(self, input_width, image_input_width, label_width, shift,
                 trainImages, trainData, trainCloud, trainAverage, trainY,
                 valImage, valData, valCloud, valAverage, valY,
                 testImage, testData, testCloud, testAverage, testY,
                 batch_size=32, label_columns=None, samples_per_day=None,
                 using_timestamp_data=False, using_shuffle=parameter.is_using_shuffle):

        # Work out the window parameters.
        self.input_width = input_width
        self.image_input_width = image_input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.using_timestamp_data = using_timestamp_data
        self.using_shuffle = using_shuffle
        self.total_window_size = self.input_width + self.label_width + self.shift
        self.samples_per_day = samples_per_day
        self.is_sampling_within_day = True if self.total_window_size <= self.samples_per_day else False

        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.image_slice = slice(self.input_width - self.image_input_width, self.input_width)
        self.image_input_indices = np.arange(self.total_window_size)[self.image_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        # Store the raw data.
        self.trainImagesX = trainImages
        self.trainDataX = trainData
        self.trainCloudX = trainCloud
        self.trainAverageX = trainAverage
        self.trainY_nor = trainY

        self.valImageX = valImage
        self.valDataX = valData
        self.valCloudX = valCloud
        self.valAverageX = valAverage
        self.valY_nor = valY

        self.testImagesX = testImage
        self.testDataX = testData
        self.testCloudX = testCloud
        self.testAverageX = testAverage
        self.testY = testY

        if parameter.dynamic_model == "one" and parameter.addAverage == True:
            self.trainCloudX = pd.concat([trainCloud, trainAverage], axis=1)
            self.valCloudX = pd.concat([valCloud, valAverage], axis=1)
            self.testCloudX = pd.concat([testCloud, testAverage], axis=1)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Image indices: {self.image_input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def plotPredictUnit(self, model, dataset, datamode):
        all_y = None
        all_pred = None
        for inputs, targets in dataset.as_numpy_iterator():
            # print("@@@@@@@@@@@@@@@@@@@@@@",inputs.shape)
            # inputs, targets = batch
            # print(inputs)
            # print("$targets.shape$", targets.shape)
            # if isinstance(inputs, tuple):
            #     for data in inputs:
            #         print
            # else:
            #     pass
            # if datamode == "data":
            #     if parameter.addAverage:
            #         data, average = inputs
            #         print("$combined  data prediction inputs shape", data.shape)
            #         print("$combined average prediction inputs shape", average.shape)
            #     else:
            #         data = inputs
            #         print("$combined  data prediction inputs shape", data.shape)
            # elif datamode == "combined":
            #     if parameter.addAverage:
            #         image, data, average = inputs
            #         print("$combined image prediction inputs shape", image.shape)
            #         print("$combined  data prediction inputs shape", data.shape)
            #         print("$combined average prediction inputs shape", average.shape)
            #     else:
            #         image, data = inputs
            #         print("$combined image prediction inputs shape", image.shape)
            #         print("$combined  data prediction inputs shape", data.shape)
            targets = targets.reshape((-1, targets.shape[-1]))
            # print("$targets.shape$", targets.shape)
            if all_y is None:
                all_y = targets
            else:
                all_y = np.vstack((all_y, targets))
            # print(all_y.shape)
            pred = model.predict(inputs)
            pred = pred.reshape((-1, pred.shape[-1]))
            # print("#pred.shape#", pred.shape)
            if all_pred is None:
                all_pred = pred
            else:
                all_pred = np.vstack((all_pred, pred))
            # all_pred = model.predict(all_input)
        # print(">>>>>>>>",all_y.shape)
        # print("!!!!!!!!",all_pred.shape)
        return all_pred, all_y

    def image_split_window(self, features):
        # print("##########",features)
        inputs = features[:, self.image_slice, :, :, :]
        # print("#####################",inputs)       #shape=(None, None, 48, 64, 3)
        # labels = la_features[:, labels_slice]
        inputs.set_shape([None, self.image_input_width, None, None, None])
        # print("#####################",inputs)       #shape=(None, 1, 48, 64, 3),
        if parameter.squeeze == True:
            inputs = tf.squeeze(inputs, axis=1)
        # print("#####################",inputs)       #shape=(None, 48, 64, 3),
        # labels.set_shape([None, label_width])

        return inputs

    def data_split_window(self, features):
        # print("##########",features)
        inputs = features[:, self.input_slice, :]
        # labels = la_features[:, labels_slice]
        # print(inputs)
        inputs.set_shape([None, self.input_width, None])
        # print(inputs)
        # labels.set_shape([None, label_width])
        return inputs

    def label_split_window(self, features):
        # print("##########",features)
        # inputs = features[:, input_slice, :, :, :]
        labels = features[:, self.labels_slice, :]
        labels.set_shape([None, self.label_width, None])
        # labels.set_shape([None, self.label_width, None])
        # if (self.takeMode==0):
        #     labels = labels
        # elif (self.takeMode==1):
        #     labels = labels[:, -1, :]
        return labels

    def input_dataset(self, data, label, cloudData=None, image=None,
                      is_timestamp_as_data=False, ganIndex=False,
                      sequence_stride=parameter.label_width, use_shuffle=False):
        ds_t, ds_u, ds_c, ds_v, ds_d = None, None, None, None, None
        rows_counter = 0
        if ganIndex:
            data = pd.DataFrame(data.index.values.astype(np.int64), index=data.index, columns=['timestamp'])
            # data = data[['timestamp']]
            label = pd.DataFrame(label.index.values.astype(np.int64), index=label.index, columns=['timestamp'])
            # label = label[['timestamp']]

        for date in np.unique(data.index.date):
            num_of_rows = len(data[data.index.date == date].index)
            data_on_date = data[
                data.index.date == date] if parameter.between8_17 and self.is_sampling_within_day else data
            label_on_date = label[
                label.index.date == date] if parameter.between8_17 and self.is_sampling_within_day else label
            data_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data_on_date,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=sequence_stride,
                shuffle=parameter.dynamic_suffle).map(self.data_split_window).unbatch()
            ds_u = data_dataset if ds_u is None else ds_u.concatenate(data_dataset)
            label_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=label_on_date,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=sequence_stride,
                shuffle=parameter.dynamic_suffle).map(self.label_split_window).unbatch()
            ds_v = label_dataset if ds_v is None else ds_v.concatenate(label_dataset)
            if cloudData is not None:
                cloudData_on_date = cloudData[
                    cloudData.index.date == date] if parameter.between8_17 and self.is_sampling_within_day else cloudData
                cloud_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                    data=cloudData_on_date,
                    targets=None,
                    sequence_length=self.total_window_size,
                    sequence_stride=sequence_stride,
                    shuffle=parameter.dynamic_suffle).map(self.data_split_window).unbatch()
                ds_c = cloud_dataset if ds_c is None else ds_c.concatenate(cloud_dataset)
            if image is not None:
                images_on_date = image[
                                 rows_counter:rows_counter + num_of_rows] if parameter.between8_17 and self.is_sampling_within_day else image
                images_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                    data=images_on_date,
                    targets=None,
                    sequence_length=self.total_window_size,
                    sequence_stride=sequence_stride,
                    shuffle=parameter.dynamic_suffle).map(self.image_split_window).unbatch()
                ds_t = images_dataset if ds_t is None else ds_t.concatenate(images_dataset)
            if is_timestamp_as_data:
                timestamps = pd.DataFrame(data_on_date.index.values, index=data_on_date.index, columns=['timestamp']) \
                    if parameter.between8_17 and self.is_sampling_within_day else pd.DataFrame(data.index.values,
                                                                                               index=data.index,
                                                                                               columns=['timestamp'])
                timestamps['month'] = timestamps['timestamp'].apply(lambda x: x.month - 1)
                timestamps['day'] = timestamps['timestamp'].apply(lambda x: x.day - 1)
                timestamps['hour'] = timestamps['timestamp'].apply(lambda x: x.hour)
                timestamps['minute'] = timestamps['timestamp'].apply(lambda x: x.minute)
                timestamps = timestamps.drop(['timestamp'], axis=1)
                datetime_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                    data=timestamps,
                    targets=None,
                    sequence_length=self.total_window_size,
                    sequence_stride=sequence_stride,
                    shuffle=parameter.dynamic_suffle).unbatch()
                ds_d = datetime_dataset if ds_d is None else ds_d.concatenate(datetime_dataset)
            if not self.is_sampling_within_day or not parameter.between8_17:
                # this means there is no need to fix the discontinuous sequence problem
                # which might happen when the data is not in continuous manner
                break
            rows_counter += num_of_rows
        data_tuple = tuple([ele for ele in [ds_t, ds_u, ds_c, ds_d] if ele is not None])
        data_tuple = data_tuple if len(data_tuple) > 1 else data_tuple[0]
        c = tf.data.Dataset.zip(data_tuple)
        if ganIndex:
            c = ds_u
        c = tf.data.Dataset.zip((c, ds_v))
        if use_shuffle:
            c = c.shuffle(self.batch_size)
        c = c.batch(self.batch_size)
        print(c)
        return c

    ##################################################################################################
    def cloud_image_split_window(self, features, cloud, allowed_labels=tf.constant([0.])):
        # print("##########",features)
        inputs = features[:, self.image_slice, :, :, :]
        # print("#####@@@@@@#####",inputs.shape)       #shape=(None, None, 48, 64, 3)
        inputs.set_shape([None, self.image_input_width, None, None, None])
        # print("#####@@@@@@#####",inputs.shape)       #shape=(None, 1, 48, 64, 3)
        cloud_label = cloud[:, self.input_slice, :]
        # print("########################",cloud_label)
        cloud_label = cloud_label[:, -1, :]
        # print("########################",cloud_label)

        isallowed = tf.equal(allowed_labels, tf.cast(cloud_label, tf.float32))
        isallowed = tf.reshape(isallowed, [-1])
        inputs = inputs[isallowed]
        return inputs

    def cloud_data_split_window(self, features, cloud, allowed_labels=tf.constant([0.])):
        # print("##########",features)
        inputs = features[:, self.input_slice, :]
        inputs.set_shape([None, self.input_width, None])
        cloud_label = cloud[:, self.input_slice, :]
        cloud_label = cloud_label[:, -1, :]
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>",cloud_label)
        isallowed = tf.equal(allowed_labels, tf.cast(cloud_label, tf.float32))
        isallowed = tf.reshape(isallowed, [-1])
        inputs = inputs[isallowed]

        return inputs

    def cloud_label_split_window(self, features, cloud, allowed_labels=tf.constant([0.])):
        # print("##########",features)
        # inputs = features[:, input_slice, :, :, :]
        labels = features[:, self.labels_slice, :]
        labels.set_shape([None, self.label_width, None])
        cloud_label = cloud[:, self.input_slice, :]
        # cloud_label = tf.reduce_sum(cloud_label,axis=1)
        cloud_label = cloud_label[:, -1, :]
        # cloud_label.set_shape([None, self.input_width, None])
        # print("&&&&&&&",labels)
        isallowed = tf.equal(allowed_labels, tf.cast(cloud_label, tf.float32))
        # print("******",isallowed)
        isallowed = tf.reshape(isallowed, [-1])
        # print("******",isallowed)
        labels = labels[isallowed]
        # print("&&&&&&&",labels)
        return labels

    def input_dataset_cloud(self, image, data, cloudData, average, label, sepMode="all", ganIndex=False):
        if ganIndex:
            data = data.index.values.astype(int).reshape((-1, 1))  # to numric dtype for convert tensor
            label = label.index.values.astype(int).reshape((-1, 1))
        if sepMode == "cloudA":
            allowed_labels = tf.constant([1.])
        elif sepMode == "cloudC":
            allowed_labels = tf.constant([0.])
        print("::::::", allowed_labels)
        cloudData = pd.DataFrame(cloudData)
        average = pd.DataFrame(average)
        # print("^^^^^^",data.shape)
        ds_c = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=cloudData,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=self.batch_size)

        ds_t = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=image,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=self.batch_size)
        ds_t = tf.data.Dataset.zip((ds_t, ds_c))  ######
        ds_t = ds_t.map(
            lambda x, cloud: self.cloud_image_split_window(x, cloud, allowed_labels=allowed_labels))  #######
        # print(ds_t)
        ds_u = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=self.batch_size)
        # print(ds_u)
        ds_u = tf.data.Dataset.zip((ds_u, ds_c))  #####
        ds_u = ds_u.map(lambda x, cloud: self.cloud_data_split_window(x, cloud, allowed_labels=allowed_labels))  #######
        # for element in ds_u:
        #     print(element)
        # print(ds_u)
        ###############################################################
        ds_a = None
        ds_a = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=average,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=self.batch_size)
        ds_a = tf.data.Dataset.zip((ds_a, ds_c))  #####
        ds_a = ds_a.map(
            lambda x, cloud: self.cloud_data_split_window(x, cloud, allowed_labels=allowed_labels))  #######
        ###############################################################

        ds_v = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=label,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=self.batch_size)
        # print(ds_v)
        ds_v = tf.data.Dataset.zip((ds_v, ds_c))  ######
        ds_v = ds_v.map(lambda x, cloud: self.cloud_label_split_window(x, cloud, allowed_labels=allowed_labels))  ######

        if parameter.addAverage:
            c = tf.data.Dataset.zip((ds_t, ds_u, ds_a))  ######
            c = tf.data.Dataset.zip((c, ds_v))  ######
        else:
            c = tf.data.Dataset.zip((ds_t, ds_u))  ######
            c = tf.data.Dataset.zip((c, ds_v))  ######
        # print("ok")
        print("input_dataset_cloud function", c)
        # if ganIndex == False:  # check is for checkwindow, no use dict_windowb
        #     print("^^^^^")
        #     c = c.map(self.dict_windowData)
        # del ds_c, ds_t, ds_u, ds_a, ds_v, image, data, cloudData, average, label
        # gc.collect()
        # c = c.cache("cache/test1")
        return c

    def input_dataset_cloud_data(self, data, cloudData, average, label, sepMode="all", ganIndex=False):
        if ganIndex:
            data = data.index.values.astype(int).reshape((-1, 1))  # to numric dtype for convert tensor
            label = label.index.values.astype(int).reshape((-1, 1))
        if sepMode == "cloudA":
            allowed_labels = tf.constant([1.])
        elif sepMode == "cloudC":
            allowed_labels = tf.constant([0.])
        print("::::::", allowed_labels)
        cloudData = pd.DataFrame(cloudData)
        average = pd.DataFrame(average)
        ds_c = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=cloudData,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=self.batch_size)

        ds_u = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=self.batch_size)
        # print(ds_u)
        ds_u = tf.data.Dataset.zip((ds_u, ds_c))
        ds_u = ds_u.map(lambda x, cloud: self.cloud_data_split_window(x, cloud, allowed_labels=allowed_labels))
        # print(ds_u)
        ###############################################################
        ds_a = None
        ds_a = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=average,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=self.batch_size)
        ds_a = tf.data.Dataset.zip((ds_a, ds_c))  #####
        ds_a = ds_a.map(
            lambda x, cloud: self.cloud_data_split_window(x, cloud, allowed_labels=allowed_labels))  #######
        ###############################################################
        ds_v = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=label,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=self.batch_size)
        # print(ds_v)
        ds_v = tf.data.Dataset.zip((ds_v, ds_c))
        ds_v = ds_v.map(lambda x, cloud: self.cloud_label_split_window(x, cloud, allowed_labels=allowed_labels))
        # print(ds_v)

        if parameter.addAverage:
            c = tf.data.Dataset.zip((ds_u, ds_a))
            c = tf.data.Dataset.zip((c, ds_v))
        else:
            c = tf.data.Dataset.zip((ds_u, ds_v))
        print("ok")
        print("input_dataset_cloud_data function", c)
        # if ganIndex == False:  # check is for checkwindow, no use dict_windowb
        #     print("^^^^^")
        #     c = c.map(self.dict_windowData)
        # del ds_c, ds_u, ds_a, ds_v, data, cloudData, average, label
        # gc.collect()
        # c = c.cache("cache/test2")
        return c

    def trainData(self, sepMode: str = "all", addcloud=False):
        # return self.input_dataset_2(self.trainDataX, self.trainCloudX, self.trainY_nor, addcloud=addcloud,
        #                             sequence_stride=1, use_shuffle=parameter.is_using_shuffle)
        if self.is_sampling_within_day or parameter.input_days is None:
            if addcloud:
                return self.input_dataset(self.trainDataX, self.trainY_nor, cloudData=self.trainCloudX,
                                          is_timestamp_as_data=self.using_timestamp_data,
                                          sequence_stride=1, use_shuffle=self.using_shuffle)
            else:
                return self.input_dataset(self.trainDataX, self.trainY_nor,
                                          is_timestamp_as_data=self.using_timestamp_data,
                                          sequence_stride=1, use_shuffle=self.using_shuffle)
        else:
            if addcloud:
                return self.input_dataset(self.trainDataX, self.trainY_nor, cloudData=self.trainCloudX,
                                          is_timestamp_as_data=self.using_timestamp_data,
                                          sequence_stride=self.samples_per_day, use_shuffle=self.using_shuffle)
            else:
                return self.input_dataset(self.trainDataX, self.trainY_nor,
                                          is_timestamp_as_data=self.using_timestamp_data,
                                          sequence_stride=self.samples_per_day, use_shuffle=self.using_shuffle)

    def valData(self, sepMode: str = "all", addcloud=False):
        # return self.input_dataset_2(self.valDataX, self.valCloudX, self.valY_nor, addcloud=addcloud, sequence_stride=1)
        if self.is_sampling_within_day or parameter.input_days is None:
            if addcloud:
                return self.input_dataset(self.valDataX, self.valY_nor, cloudData=self.valCloudX,
                                          is_timestamp_as_data=self.using_timestamp_data,
                                          sequence_stride=1)
            else:
                return self.input_dataset(self.valDataX, self.valY_nor, is_timestamp_as_data=self.using_timestamp_data,
                                          sequence_stride=1)
        else:
            if addcloud:
                return self.input_dataset(self.valDataX, self.valY_nor, cloudData=self.valCloudX,
                                          is_timestamp_as_data=self.using_timestamp_data,
                                          sequence_stride=self.samples_per_day)
            else:
                return self.input_dataset(self.valDataX, self.valY_nor, is_timestamp_as_data=self.using_timestamp_data,
                                          sequence_stride=self.samples_per_day)

    def testData(self, sepMode: str = "all", ganIndex=False, addcloud=False):
        # return self.input_dataset_2(self.testDataX, self.testCloudX, self.testY, ganIndex=ganIndex, addcloud=addcloud,
        #                             sequence_stride=parameter.label_width)
        if addcloud:
            return self.input_dataset(self.testDataX, self.testY, cloudData=self.testCloudX,
                                      is_timestamp_as_data=self.using_timestamp_data, ganIndex=ganIndex,
                                      sequence_stride=self.label_width)
        else:
            return self.input_dataset(self.testDataX, self.testY, is_timestamp_as_data=self.using_timestamp_data,
                                      ganIndex=ganIndex,
                                      sequence_stride=self.label_width)
        ############################################################################################################################################

    def train(self, sepMode: str = "all", addcloud=False):
        if addcloud:
            return self.input_dataset(self.trainDataX, self.trainY_nor, cloudData=self.trainCloudX,
                                      image=self.trainImagesX, is_timestamp_as_data=self.using_timestamp_data,
                                      sequence_stride=1, use_shuffle=self.using_shuffle)
        else:
            return self.input_dataset(self.trainDataX, self.trainY_nor, image=self.trainImagesX,
                                      is_timestamp_as_data=self.using_timestamp_data,
                                      sequence_stride=1, use_shuffle=self.using_shuffle)

    def val(self, sepMode: str = "all", addcloud=False):
        if addcloud:
            return self.input_dataset(self.valDataX, self.valY_nor, cloudData=self.valCloudX, image=self.valImageX,
                                      is_timestamp_as_data=self.using_timestamp_data,
                                      sequence_stride=1)
        else:
            return self.input_dataset(self.valDataX, self.valY_nor, image=self.valImageX,
                                      is_timestamp_as_data=self.using_timestamp_data,
                                      sequence_stride=1)

    def test(self, sepMode: str = "all", ganIndex=False, addcloud=False):
        if addcloud:
            return self.input_dataset(self.testDataX, self.testY, cloudData=self.testCloudX, image=self.testImagesX,
                                      is_timestamp_as_data=self.using_timestamp_data,
                                      ganIndex=ganIndex,
                                      sequence_stride=parameter.label_width, use_shuffle=False)
        else:
            return self.input_dataset(self.testDataX, self.testY, image=self.testImagesX,
                                      is_timestamp_as_data=self.using_timestamp_data,
                                      ganIndex=ganIndex,
                                      sequence_stride=parameter.label_width, use_shuffle=False)

    ##############################################################################################################################################
    ## two model
    ##############################################################################################################################################
    def trainDataAC(self, sepMode: str = "all"):
        return self.input_dataset_cloud_data(self.trainDataX, self.trainCloudX, self.trainAverageX, self.trainY_nor,
                                             sepMode=sepMode)

    def valDataAC(self, sepMode: str = "all"):
        return self.input_dataset_cloud_data(self.valDataX, self.valCloudX, self.valAverageX, self.valY_nor,
                                             sepMode=sepMode)

    def testDataAC(self, sepMode: str = "all", ganIndex=False):
        return self.input_dataset_cloud_data(self.testDataX, self.testCloudX, self.testAverageX, self.testY,
                                             sepMode=sepMode, ganIndex=ganIndex)
        ########################################################################################################################################

    def trainAC(self, sepMode: str = "all"):  # 因為conv3D內的Maxpooling3D假設batch內數量為0會有GPU問題(其他2D的不會有問題)，所以在這裡把0的挑出來刪掉重組
        c = self.input_dataset_cloud(self.trainImagesX, self.trainDataX, self.trainCloudX, self.trainAverageX,
                                     self.trainY_nor, sepMode=sepMode)
        all_y = None
        all_input0, all_input1, all_input2 = None, None, None
        # inputs0, inputs1 = None, None
        if parameter.addAverage:
            for inputs, targets in c.as_numpy_iterator():
                image, data, average = inputs
                print("$train combined image prediction inputs shape", image.shape)
                print("$train combined data prediction inputs shape", data.shape)
                print("$train combined average prediction inputs shape", average.shape)

                if data.shape[0] != 0:
                    if all_y is None:
                        all_y = targets
                    else:
                        all_y = np.vstack((all_y, targets))
                    # print(all_y.shape) 
                    # print("#pred.shape#", pred.shape)
                    # print(inputs)
                    if all_input0 is None:
                        all_input0 = inputs[0]
                        all_input1 = inputs[1]
                        all_input2 = inputs[2]
                    else:
                        print(all_input0.shape)
                        print(all_input1.shape)
                        print(all_input2.shape)
                        print(inputs[0].shape)
                        print(inputs[1].shape)
                        print(inputs[2].shape)
                        all_input0 = tf.concat([all_input0, inputs[0]], axis=0)
                        all_input1 = tf.concat([all_input1, inputs[1]], axis=0)
                        all_input2 = tf.concat([all_input2, inputs[2]], axis=0)
                        print(all_input0.shape)
                        print(all_input1.shape)
                        print(all_input2.shape)
                        # inputs0 = np.vstack(all_input0, inputs[0])
                        # inputs1 = np.vstack(all_input1, inputs[1])
            print("$$input0$$", all_input0.shape)
            print("$$input1$$", all_input1.shape)
            print("$$input2$$", all_input2.shape)
            all_input = all_input0, all_input1, all_input2  ######

        else:
            for inputs, targets in c.as_numpy_iterator():
                image, data = inputs
                print("$train combined image prediction inputs shape", image.shape)
                print("$train combined data prediction inputs shape", data.shape)

                if data.shape[0] != 0:
                    if all_y is None:
                        all_y = targets
                    else:
                        all_y = np.vstack((all_y, targets))
                    # print(all_y.shape) 
                    # print("#pred.shape#", pred.shape)
                    # print(inputs)
                    if all_input0 is None:
                        all_input0 = inputs[0]
                        all_input1 = inputs[1]
                    else:
                        print(all_input0.shape)
                        print(all_input1.shape)
                        print(inputs[0].shape)
                        print(inputs[1].shape)
                        all_input0 = tf.concat([all_input0, inputs[0]], axis=0)
                        all_input1 = tf.concat([all_input1, inputs[1]], axis=0)
                        print(all_input0.shape)
                        print(all_input1.shape)
            print("$$input0$$", all_input0.shape)
            print("$$input1$$", all_input1.shape)
            all_input = all_input0, all_input1

        c = tf.data.Dataset.from_tensor_slices((all_input, all_y))
        c = c.batch(200)
        print("&&&&&&", c)
        c = c.cache("cache/test1")
        return c
        # return self.input_dataset_cloud(self.trainImagesX, self.trainDataX, self.trainCloudX, self.trainY_nor, sepMode=sepMode)

    def valAC(self, sepMode: str = "all"):
        return self.input_dataset_cloud(self.valImageX, self.valDataX, self.valCloudX, self.valAverageX, self.valY_nor,
                                        sepMode=sepMode)

    def testAC(self, sepMode: str = "all", ganIndex=False):
        return self.input_dataset_cloud(self.testImagesX, self.testDataX, self.testCloudX, self.testAverageX,
                                        self.testY, sepMode=sepMode, ganIndex=ganIndex)

    ########################################################################################################################################

    '''def input_dataset(self, data, label, ganIndex=False):        
        ds_u = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=parameter.batchsize)
        ds_u = ds_u.map(self.data_split_window)
        
        ds_v = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=label,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=parameter.dynamic_suffle,
            batch_size=parameter.batchsize)
        # print(ds_v)
        ds_v = ds_v.map(self.label_split_window)

        c = tf.data.Dataset.zip((ds_u, ds_v))
        return c'''

    def checkWindow(self):
        temp_trainData = pd.DataFrame(data={self.label_columns[0]: np.arange(0, len(self.trainDataX))})
        temp_cloudData = pd.DataFrame(data={self.label_columns[0]: np.arange(0, len(self.trainCloudX))})
        temp_trainImage = pd.DataFrame(data={self.label_columns[0]: np.arange(0, len(self.trainImagesX))})
        temp_trainLabel = pd.DataFrame(data={self.label_columns[0]: np.arange(0, len(self.trainY_nor))})

        for inputs, labels in self.input_dataset_2(temp_trainData, temp_cloudData, temp_trainLabel).take(2):
            # for inputs, labels in self.train.take(2):
            #     inputs = inputs[:,:,0]
            # labels = labels[:, :, 0]
            inputs = inputs
            labels = labels
            log = logging.getLogger(parameter.experient_label)
            log.debug((inputs.numpy().shape, "=>", labels.numpy().shape))
            log.debug("1st batch in batches-----------------------")
            log.debug((inputs.numpy()[0, :].tolist(), "\n=>", labels.numpy()[0, :].tolist()))
            if (inputs.numpy().shape[0] > 1):
                log.debug("2ed batch in batches-----------------------")
                log.debug((inputs.numpy()[1, :].tolist(), "\n=>", labels.numpy()[1, :].tolist()))
            log.debug("last batch in batches-----------------------")
            log.debug((inputs.numpy()[-1, :].tolist(), "\n=>", labels.numpy()[-1, :].tolist()))
            log.debug("NEXT TACK=========================================================")

    # 產生label資料的 時間timeindex
    def getlabelDataIndex(self, index_dataset, name=None):
        all_y_index = None
        all_x_index = None
        for inputs, label in index_dataset.as_numpy_iterator():
            if name != "Persistence" and name != "MA":
                inputs, _ = inputs
            if all_x_index is None:
                # print(inputs)
                all_x_index = inputs
            else:
                all_x_index = np.vstack((all_x_index, inputs))
            if all_y_index is None:
                all_y_index = label
            else:
                all_y_index = np.vstack((all_y_index, label))
                # all_y_index = np.stack((all_y_index, label), axis=0)
        all_x_index = all_x_index.flatten().astype("datetime64[ns]")
        all_y_index = all_y_index.flatten().astype("datetime64[ns]")
        return all_x_index, all_y_index

    # 產生label資料的 時間timeindex
    def getlabelDataIndex_withImage(self, index_dataset, name=None):
        all_y_index = None
        all_x_index = None
        for inputs, label in index_dataset.as_numpy_iterator():
            if all_x_index is None:
                # print(inputs)
                all_x_index = inputs[1]
            else:
                all_x_index = np.vstack((all_x_index, inputs[1]))
            if all_y_index is None:
                all_y_index = label
            else:
                all_y_index = np.vstack((all_y_index, label))
                # all_y_index = np.stack((all_y_index, label), axis=0)
        all_x_index = all_x_index.flatten().astype("datetime64[ns]")
        all_y_index = all_y_index.flatten().astype("datetime64[ns]")
        return all_x_index, all_y_index

    # 產生label資料的 時間timeindex
    def getlabelDataIndex_withData(self, index_dataset, name=None):
        all_y_index = None
        all_x_index = None
        print(">>>>>>>>", index_dataset)
        for inputs, label in index_dataset.as_numpy_iterator():
            if parameter.addAverage:
                inputs, _ = inputs
            if all_x_index is None:
                # print(inputs)
                all_x_index = inputs
            else:
                all_x_index = np.vstack((all_x_index, inputs))
            if all_y_index is None:
                all_y_index = label
            else:
                all_y_index = np.vstack((all_y_index, label))
                # all_y_index = np.stack((all_y_index, label), axis=0)
        all_x_index = all_x_index[:, -5:, :].flatten().astype("datetime64[ns]")
        all_y_index = all_y_index.flatten().astype("datetime64[ns]")
        return all_x_index, all_y_index

    def allPlot(self, model=None, plot_col_index=0, name="Example", scaler=None,
                save_csv=parameter.save_csv, save_plot=parameter.save_plot,
                rainSepMode=False, datamode="data"):
        pattern = "[" + "|\'\"" + "]"
        output_filelabel = "plot/{}/{}".format(parameter.experient_label, name)
        output_filelabel = re.sub(pattern, "", output_filelabel)
        cloudA_label_index = None
        cloudC_label_index = None
        log = logging.getLogger(parameter.experient_label)
        if len(model) == 1:
            if datamode == "data":  # persistence, moving average
                all_pred, all_y = self.plotPredictUnit(model[0], self.testData(sepMode="all", ganIndex=False,
                                                                               addcloud=parameter.addAverage),
                                                       datamode=datamode)
                _, all_y_index = self.getlabelDataIndex_withData(
                    self.testData(sepMode="all", ganIndex=True, addcloud=parameter.addAverage), name)
            elif datamode == "combined":
                all_pred, all_y = self.plotPredictUnit(model[0], self.test(sepMode="all", ganIndex=False,
                                                                           addcloud=parameter.addAverage),
                                                       datamode=datamode)
                _, all_y_index = self.getlabelDataIndex_withImage(
                    self.test(sepMode="all", ganIndex=True, addcloud=parameter.addAverage), name)
        elif len(model) == 2:
            if datamode == "data":  # cnnlstm
                cloudA_pred, cloudA_y = self.plotPredictUnit(model[0],
                                                             self.testDataAC(sepMode="cloudA", ganIndex=False),
                                                             datamode=datamode)
                cloudC_pred, cloudC_y = self.plotPredictUnit(model[1],
                                                             self.testDataAC(sepMode="cloudC", ganIndex=False),
                                                             datamode=datamode)

                _, cloudA_label_index = self.getlabelDataIndex_withData(
                    self.testDataAC(sepMode="cloudA", ganIndex=True), name)
                _, cloudC_label_index = self.getlabelDataIndex_withData(
                    self.testDataAC(sepMode="cloudC", ganIndex=True), name)

                if cloudC_y is None:  # todo if cloudC data not enough
                    # print(cloudA_y.shape)
                    log.info("C is none and cloudA_y shape: {}".format(cloudA_y.shape))
                    # print("####################################")
                    all_pred, all_y = self.plotPredictUnit(model[0], self.testDataAC(sepMode="cloudA", ganIndex=False),
                                                           name=name)
                    _, all_y_index = self.getlabelDataIndex_withData(self.testDataAC(sepMode="cloudA", ganIndex=True),
                                                                     name)
                elif cloudA_y is None:  # todo if cloudA data not enough
                    # print(cloudC_y.shape)
                    log.info("A is none and cloudC_y shape: {}".format(cloudC_y.shape))
                    # print("####################################")
                    all_pred, all_y = self.plotPredictUnit(model[1], self.testDataAC(sepMode="cloudC", ganIndex=False),
                                                           datamode=datamode)
                    _, all_y_index = self.getlabelDataIndex_withData(self.testDataAC(sepMode="cloudC", ganIndex=True),
                                                                     name)
                else:
                    # print(cloudC_y.shape)
                    # print(cloudA_y.shape)
                    # print(cloudC_label_index.shape)
                    # print(cloudA_label_index.shape)
                    log.info("cloudA_y shape: {}".format(cloudA_y.shape))
                    log.info("cloudC_y shape: {}".format(cloudC_y.shape))
                    # print("####################################")
                    all_pred = np.vstack((cloudA_pred, cloudC_pred))
                    all_y = np.vstack((cloudA_y, cloudC_y))
                    all_y_index = np.vstack((cloudA_label_index.reshape((-1, 1)), cloudC_label_index.reshape((-1, 1))))

            elif datamode == "combined":  # conv3D_c_cnnlstm
                cloudA_pred, cloudA_y = self.plotPredictUnit(model[0], self.testAC(sepMode="cloudA", ganIndex=False),
                                                             datamode=datamode)
                cloudC_pred, cloudC_y = self.plotPredictUnit(model[1], self.testAC(sepMode="cloudC", ganIndex=False),
                                                             datamode=datamode)
                # print(cloudC_y.shape)
                # print(cloudA_y.shape)

                _, cloudA_label_index = self.getlabelDataIndex_withImage(self.testAC(sepMode="cloudA", ganIndex=True),
                                                                         name)
                _, cloudC_label_index = self.getlabelDataIndex_withImage(self.testAC(sepMode="cloudC", ganIndex=True),
                                                                         name)

                if cloudC_y is None:  # todo if cloudC data not enough
                    # print(cloudA_y.shape)
                    log.info("C is none and cloudA_y shape: {}".format(cloudA_y.shape))
                    # print("####################################")
                    all_pred, all_y = self.plotPredictUnit(model[0], self.testAC(sepMode="cloudA", ganIndex=False),
                                                           datamode=datamode)
                    _, all_y_index = self.getlabelDataIndex_withImage(self.testAC(sepMode="cloudA", ganIndex=True),
                                                                      name)
                elif cloudA_y is None:  # todo if cloudA data not enough
                    # print(cloudC_y.shape)
                    log.info("A is none and cloudC_y shape: {}".format(cloudC_y.shape))
                    # print("####################################")
                    all_pred, all_y = self.plotPredictUnit(model[1], self.testAC(sepMode="cloudC", ganIndex=False),
                                                           datamode=datamode)
                    _, all_y_index = self.getlabelDataIndex_withImage(self.testAC(sepMode="cloudC", ganIndex=True),
                                                                      name)
                else:
                    # print(cloudC_y.shape)
                    # print(cloudA_y.shape)
                    # print(cloudC_label_index.shape)
                    # print(cloudA_label_index.shape)
                    log.info("cloudA_y shape: {}".format(cloudA_y.shape))
                    log.info("cloudC_y shape: {}".format(cloudC_y.shape))
                    # print("####################################")
                    all_pred = np.vstack((cloudA_pred, cloudC_pred))
                    all_y = np.vstack((cloudA_y, cloudC_y))
                    all_y_index = np.vstack((cloudA_label_index.reshape((-1, 1)), cloudC_label_index.reshape((-1, 1))))
            # log.info("cloudA_y shape: {}".format(cloudA_y.shape))
            # log.info("cloudC_y shape: {}".format(cloudC_y.shape))
            log.info("all_pred shape: {}".format(all_pred.shape))
            log.info("all_y_index shape: {}".format(all_y_index.shape))
            log.info("all_y shape: {}".format(all_y.shape))
            # all_pred = np.vstack((cloudA_pred, cloudC_pred))
            # all_y = np.vstack((cloudA_y, cloudC_y))
            # all_y_index = np.vstack((cloudA_label_index.reshape((-1, 1)), cloudC_label_index.reshape((-1, 1))))        
        ## inverse_sacler
        if scaler is not None:
            all_y = scaler.inverse_transform(all_y)
            all_pred = scaler.inverse_transform(all_pred)
        #######col filter
        all_y = all_y.astype(np.float32)
        all_pred = all_pred.astype(np.float32)
        print("$$$$$$$$$$$$$$$$", all_pred.shape)
        # print("$$$$$$$$$$$$$$$$", all_y_index.shape)
        # print("$$$$$$$$$$$$$$$$", all_y.shape)

        df_pred = pd.DataFrame(all_pred, columns=parameter.target, index=all_y_index.ravel()).sort_index()
        df_gt = pd.DataFrame(all_y, columns=parameter.target, index=all_y_index.ravel()).sort_index()
        # df.index = all_y_index.ravel()
        # df = df.sort_index()
        if parameter.test_between8_17:
            df_pred = df_pred.between_time(parameter.start, parameter.end)
            df_gt = df_gt.between_time(parameter.start, parameter.end)

        # post process
        tdf = [df_gt, df_pred]
        # tdf[1][tdf[1] < 0.0] = 0.0  # 預測負值 轉 0.0

        log = logging.getLogger(parameter.experient_label)
        metircs_dist = my_metrics.log_metrics(tdf, name)
        sep_metircs_dist, _ = my_metrics.seperate_log_metrics(tdf, name, self.label_width)
        metircs_dist_by_day = my_metrics.log_metrics_day_by_day(tdf, name, parameter.output_days)
        all_dict = {**metircs_dist, **sep_metircs_dist, **metircs_dist_by_day}

        # cloud_label 標籤
        try:
            os.mkdir(Path("plot/{}".format(parameter.experient_label)))
        except:
            print("plotDir exist")
        if cloudA_label_index is not None:
            df_pred['cloud_label'] = 0
            df_gt['cloud_label'] = 0
            df_pred.loc[cloudA_label_index, 'cloud_label'] = 1
            df_gt.loc[cloudA_label_index, 'cloud_label'] = 1
        ####plot all column
        if save_plot:
            pattern = "[" + "|\'\"" + "]"
            for col_name in df_pred.columns[:-1]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_pred.index,
                    y=df_pred[col_name],
                    mode='lines+markers',
                    marker_color='rgba(240, 14, 14, 1)',  # red
                    name='Predict'  # Style name/legend entry with html tags
                ))

                fig.add_trace(go.Scatter(
                    x=df_pred.index,
                    y=df_gt[col_name],
                    mode='lines+markers',
                    marker_color='rgba(44, 240, 14, 1)',  # green
                    name='GroundTruth'  # Style name/legend entry with html tags
                ))
                # 額外plot雨天預測 在同一張圖上
                if cloudC_label_index is not None:
                    # plot all , and then override color
                    temp_df = df_pred.copy()
                    temp_df.loc[temp_df.cloud_label == 1] = None
                    fig.add_scattergl(
                        x=temp_df.index,
                        y=temp_df[col_name],
                        mode='lines+markers',
                        marker_color='rgba(245, 158, 66, 1)',  # red
                        name='cloudA'  # Style name/legend entry with html tags
                    )
                    temp_df = df_pred.copy()
                    temp_df.loc[temp_df.cloud_label == 0] = None
                    fig.add_scattergl(
                        x=temp_df.index,
                        y=temp_df[col_name],
                        mode='lines+markers',
                        marker_color='rgba(66, 197, 245, 1)',  # green
                        name='cloudC'  # Style name/legend entry with html tags
                    )
                # fig.update_layout(uniformtext_minsize=30, uniformtext_mode='hide')
                fig.update_traces(textfont_size=30)
                # df = df_gt.merge(df_pred)
                ###
                plot_path = re.sub(pattern, "", output_filelabel + "_" + col_name + ".html")
                fig.write_html(str(Path(plot_path)))
        if (save_csv):
            # df.to_csv(Path(output_filelabel + ".csv"))
            df_pred.to_csv(Path(output_filelabel + "_Pred.csv"))
            df_gt.to_csv(Path(output_filelabel + "_GT.csv"))
        return all_dict
