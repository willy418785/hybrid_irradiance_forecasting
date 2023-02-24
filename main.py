# python main.py -d  ../skyImage
# import the necessary packages
import datetime

import tensorflow as tf
from pyimagesearch import datasets, model_AR, time_embedding
from pyimagesearch import models
from pyimagesearch import model_resnet
from pyimagesearch import model_solarnet
from pyimagesearch import model_convlstm
from pyimagesearch import model_conv3D
from pyimagesearch import model_cnnLSTM
from pyimagesearch import model_multiCnnLSTM
from pyimagesearch import model_3Dresnet
from pyimagesearch import model_transformer, model_convGRU, preprocess_utils
from pyimagesearch.lstnet_model import LSTNetModel
from pyimagesearch.model_baseline import Persistence, MA
from pyimagesearch.lstnet_util import GetArguments, LSTNetInit, GetArgumentsDict
from pyimagesearch.preprocess_utils import SplitInputByDay, MultipleDaysConvEmbed
from pyimagesearch import series_decomposition
from pyimagesearch import my_metrics
from pyimagesearch import Msglog
from pyimagesearch import parameter
from pyimagesearch.windowsGenerator import *
from pyimagesearch.datautil import *

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import optimizers
from tensorflow.keras.layers import concatenate
import numpy as np
import argparse
import locale
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.applications.resnet50 import ResNet50
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.83)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # =GPU使用哪一些編號的 例如"0" , "1" , "-1"(cpu)
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

def ModelTrainer(dataGnerator: WindowGenerator,
                 model,
                 generatorMode="",
                 testEpoch=0,
                 name="Example"):
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                  , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()
            , my_metrics.VWMAPE
            , my_metrics.root_relative_squared_error
            , my_metrics.corr])
    model.summary()
    tf.keras.backend.clear_session()
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    if generatorMode == "combined":
        history = model.fit(dataGnerator.train(addcloud=parameter.addAverage),
                            validation_data=dataGnerator.val(addcloud=parameter.addAverage),
                            epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        all_pred, all_y = dataGnerator.plotPredictUnit(model, dataGnerator.val(addcloud=parameter.addAverage),
                                                       datamode=generatorMode)
    # history = model.fit(dataGnerator.train(), validation_data=dataGnerator.val(),
    # 			epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
    # all_pred, all_y = dataGnerator.plotPredictUnit(model, dataGnerator.val())

    elif generatorMode == "data":
        # log_dir = "logs/fit/{}_".format(name) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(dataGnerator.trainData(addcloud=parameter.addAverage),
                            validation_data=dataGnerator.valData(addcloud=parameter.addAverage),
                            epochs=testEpoch, batch_size=parameter.batchsize,
                            callbacks=[parameter.earlystoper])  # [tensorboard_callback, parameter.earlystoper]
        all_pred, all_y = dataGnerator.plotPredictUnit(model, dataGnerator.valData(addcloud=parameter.addAverage),
                                                       datamode=generatorMode)
    # history = model.fit(dataGnerator.trainData(), validation_data=dataGnerator.valData(),
    # 			epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
    # all_pred, all_y = dataGnerator.plotPredictUnit(model, dataGnerator.valData())

    elif generatorMode == "image":
        history = model.fit(dataGnerator.trainWithArg, validation_data=dataGnerator.valWithArg,
                            epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        all_pred, all_y = dataGnerator.plotPredictUnit(model, dataGnerator.valWithArg, datamode=generatorMode)

    # test_performance = model.evaluate(dataGnerator.test)
    # print(test_performance)
    val_performance = []
    val_performance.append(mean_squared_error(all_y, all_pred))  # mse
    val_performance.append(mean_absolute_error(all_y, all_pred))  # mae
    val_performance.append(my_metrics.mean_absolute_percentage_error(all_y, all_pred))  # MAPE
    val_performance.append(my_metrics.weighted_mean_absolute_percentage_error(all_y, all_pred))  # WMAPE
    # val_performance.append(corr(gt, pred).numpy())  # CORR
    return model, val_performance


def ModelTrainer_cloud(dataGnerator: WindowGenerator,
                       model1, model2,
                       generatorMode="",
                       testEpoch=0,
                       sepMode="all",
                       name="Example"):
    model1.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                   , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()
            , my_metrics.VWMAPE
            , my_metrics.corr])
    model2.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                   , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()
            , my_metrics.VWMAPE
            , my_metrics.corr])

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    if generatorMode == "combined":
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        # objgraph.show_growth()
        '''# sep_trainA = dataGnerator.trainAC(sepMode="cloudA")
        # sep_trainC = dataGnerator.trainAC(sepMode="cloudC")
        # sep_valA = dataGnerator.valAC(sepMode="cloudA")
        # sep_valC = dataGnerator.valAC(sepMode="cloudC")
        model1.fit(sep_trainA, validation_data=sep_valA,
                    epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        A_pred, A_y = dataGnerator.plotPredictUnit(model1, sep_valA, name=name)
        tf.compat.v1.get_default_graph().finalize()

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        model2.fit(sep_trainC, validation_data=sep_valC,
                    epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        C_pred, C_y = dataGnerator.plotPredictUnit(model2, sep_valC, name=name)
        tf.compat.v1.get_default_graph().finalize()'''
        model1.fit(dataGnerator.trainAC(sepMode="cloudA"), validation_data=dataGnerator.valAC(sepMode="cloudA"),
                   epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        A_pred, A_y = dataGnerator.plotPredictUnit(model1, dataGnerator.valAC(sepMode="cloudA"), datamode=generatorMode)

        model2.fit(dataGnerator.trainAC(sepMode="cloudC"), validation_data=dataGnerator.valAC(sepMode="cloudC"),
                   epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        C_pred, C_y = dataGnerator.plotPredictUnit(model2, dataGnerator.valAC(sepMode="cloudC"), datamode=generatorMode)
    # objgraph.show_growth()
    elif generatorMode == "data":
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        '''# sep_trainA = dataGnerator.trainDataAC(sepMode="cloudA")
        # sep_trainC = dataGnerator.trainDataAC(sepMode="cloudC")
        # sep_valA = dataGnerator.valDataAC(sepMode="cloudA")
        # sep_valC = dataGnerator.valDataAC(sepMode="cloudC")
        model1.fit(sep_trainA, validation_data=sep_valA,
                    epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        A_pred, A_y = dataGnerator.plotPredictUnit(model1, sep_valA, name=name)
        # print(A_pred, A_y)
        
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        model2.fit(sep_trainC, validation_data=sep_valC,
                    epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        C_pred, C_y = dataGnerator.plotPredictUnit(model2, sep_valC, name=name)
        # print(C_pred, C_y)'''
        model1.fit(dataGnerator.trainDataAC(sepMode="cloudA"), validation_data=dataGnerator.valDataAC(sepMode="cloudA"),
                   epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        A_pred, A_y = dataGnerator.plotPredictUnit(model1, dataGnerator.valDataAC(sepMode="cloudA"),
                                                   datamode=generatorMode)
        # print(A_pred, A_y)

        model2.fit(dataGnerator.trainDataAC(sepMode="cloudC"), validation_data=dataGnerator.valDataAC(sepMode="cloudC"),
                   epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        C_pred, C_y = dataGnerator.plotPredictUnit(model2, dataGnerator.valDataAC(sepMode="cloudC"),
                                                   datamode=generatorMode)
    # print(C_pred, C_y)

    val_performance = []
    if A_pred is not None:
        val_performance.append(mean_squared_error(A_y, A_pred))  # mse
        val_performance.append(mean_absolute_error(A_y, A_pred))  # mae
        # val_performance.append(my_metrics.mean_absolute_percentage_error(A_y, A_pred))  # MAPE
        # val_performance.append(my_metrics.VWMAPE(A_y, A_pred))  # VWMAPE
        val_performance.append(my_metrics.weighted_mean_absolute_percentage_error(A_y, A_pred))  # WMAPE
    val_performance_2 = []
    if C_pred is not None:
        val_performance_2.append(mean_squared_error(C_y, C_pred))  # mse
        val_performance_2.append(mean_absolute_error(C_y, C_pred))  # mae
        # val_performance_2.append(my_metrics.mean_absolute_percentage_error(C_y, C_pred))  # MAPE
        # val_performance_2.append(my_metrics.VWMAPE(C_y, C_pred))  # VWMAPE
        val_performance_2.append(my_metrics.weighted_mean_absolute_percentage_error(C_y, C_pred))  # WMAPE
    # val_performance_2.append(corr(gt, pred).numpy())  # CORR
    return model1, val_performance, model2, val_performance_2


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--test_month", type=int, required=False, default=parameter.test_month,
                    help="test_month")
    ap.add_argument("-n", "--experient_label", type=str, required=True, default=parameter.experient_label,
                    help="experient_label")
    args = vars(ap.parse_args())
    parameter.test_month = args["test_month"]
    parameter.experient_label = args["experient_label"]
    # Initialise logging
    log = Msglog.LogInit(parameter.experient_label, "logs/{}".format(parameter.experient_label), 10, True, True)

    log.info("Python version: %s", sys.version)
    log.info("Tensorflow version: %s", tf.__version__)
    log.info("Keras version: %s ... Using tensorflow embedded keras", tf.keras.__version__)

    # log.info("static_suffle: {}".format(parameter.static_suffle))
    # log.info("dynamic_suffle: {}".format(parameter.dynamic_suffle))
    log.info("timeseries: {}".format(parameter.timeseries))
    log.info("input features: {}".format(parameter.features))
    log.info("targets: {}".format(parameter.target))
    if parameter.is_using_image_data:
        log.info("image_input_width: {}".format(parameter.image_input_width3D))
        log.info("image_depth: {}".format(parameter.image_depth))
    log.info("after_minutes: {}".format(parameter.after_minutes))
    log.info("batchsize: {}".format(parameter.batchsize))
    log.info("early stop: {}".format(parameter.earlystoper is not None))
    log.info("model_list: {}".format(parameter.model_list))
    # log.info("class_type: {}".format(parameter.class_type))
    log.info("epoch list: {}".format(parameter.epoch_list))
    log.info("normalization: {}".format(parameter.normalization))
    log.info("split mode: {}".format(parameter.split_mode))
    log.info("test month: {}".format(parameter.test_month))
    log.info("data input: {}".format(parameter.inputs))
    log.info("Add sun_average: {}".format(parameter.addAverage))
    log.info("Model nums: {}".format(parameter.dynamic_model))
    log.info("Using shuffle: {}".format(parameter.is_using_shuffle))
    log.info("smoothing type: {}".format(parameter.smoothing_type))
    log.info("csv file name: {}".format(parameter.csv_name))
    log.info("only using daytime data: {}".format(parameter.between8_17))
    log.info("only evaluate daytime prediction: {}".format(parameter.test_between8_17))
    log.info("time granularity: {}".format(parameter.time_granularity))

    # construct the path to the input .txt file that contains information
    # on each house in the dataset and then load the dataset
    log.info("loading cloud attributes...")
    train_path = os.path.sep.join([parameter.csv_name])
    val_path = None
    test_path = None

    data_with_weather_info = DataUtil(train_path=train_path,
                                      val_path=val_path,
                                      test_path=test_path,
                                      normalise=parameter.norm_mode,
                                      label_col=parameter.target,
                                      feature_col=parameter.features,
                                      split_mode=parameter.split_mode,
                                      month_sep=parameter.test_month)

    # data_with_cloud_info = DataUtil(train_path=train_path,
    #                                 val_path=val_path,
    #                                 test_path=test_path,
    #                                 normalise=parameter.norm_mode,
    #                                 label_col=parameter.target,
    #                                 feature_col=parameter.target,
    #                                 split_mode=parameter.split_mode,
    #                                 month_sep=parameter.test_month)

    data_for_baseline = DataUtil(train_path=train_path,
                                 val_path=val_path,
                                 test_path=test_path,
                                 normalise=parameter.norm_mode,
                                 label_col=parameter.target,
                                 feature_col=parameter.target,
                                 split_mode=parameter.split_mode,
                                 month_sep=parameter.test_month)
    '''trainImages = dataUtil.load_house_images(dataUtil.train_df, parameter.datasetPath)
    train_df = dataUtil.train_df
    train_df_cloud = dataUtil.train_df_cloud######
    train_df_average = dataUtil.train_df_average######
    
    valImages = dataUtil.load_house_images(dataUtil.val_df, parameter.datasetPath)
    val_df = dataUtil.val_df
    val_df_cloud = dataUtil.val_df_cloud######
    val_df_average = dataUtil.val_df_average######
    
    testImages = dataUtil.load_house_images(dataUtil.test_df, parameter.datasetPath)
    test_df = dataUtil.test_df
    test_df_cloud = dataUtil.test_df_cloud######
    test_df_average = dataUtil.test_df_average######'''
    '''train_df, val_df, test_df, train_df_cloud, val_df_cloud, test_df_cloud, train_df_average, val_df_average, test_df_average = datasets.load_house_attributes(inputPath=train_path, month_sep=parameter.test_month, label_col=parameter.target, keep_date=True)
    print(train_df)
    print(val_df)
    print(test_df)
    trainImages = datasets.load_house_images(df=train_df, inputPath=parameter.datasetPath)
    valImages = datasets.load_house_images(df=val_df, inputPath=parameter.datasetPath)
    testImages = datasets.load_house_images(df=test_df, inputPath=parameter.datasetPath)'''
    ##########################################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################
    # windows generator#########################################################################################################################
    modelMetricsRecorder = {}
    if parameter.timeseries:
        dataUtil = data_with_weather_info
        # trainY_nor = np.expand_dims(trainY_nor, axis=-1)
        # valY_nor = np.expand_dims(valY_nor, axis=-1)
        # testY = np.expand_dims(testY, axis=-1)
        if parameter.input_days is None or parameter.output_days is None:
            assert type(parameter.input_width) is int
            assert type(parameter.shifted_width) is int
            assert type(parameter.label_width) is int
            if parameter.is_using_image_data:
                assert type(parameter.image_input_width3D) is int
                image_input_width = parameter.image_input_width3D
            else:
                image_input_width = 0
            if "MA" in parameter.model_list:
                assert type(parameter.MA_width) is int
                MA_width = parameter.MA_width
            input_width = parameter.input_width
            shift = parameter.shifted_width
            label_width = parameter.label_width
            log.info("\n------In-day Prediction------")
            log.info("input width: {}".format(input_width))
            log.info("shift width: {}".format(shift))
            log.info("label width: {}".format(label_width))
            if parameter.is_using_image_data:
                log.info("images width: {}".format(image_input_width))
            if "MA" in parameter.model_list:
                log.info("MA width: {}".format(MA_width))
        else:
            assert type(parameter.input_days) is int
            assert type(parameter.output_days) is int
            assert type(parameter.shifted_days) is int
            if parameter.is_using_image_data:
                assert type(parameter.image_input_width3D) is int
                image_input_width = parameter.image_input_width3D
            else:
                image_input_width = 0
            if "MA" in parameter.model_list:
                assert type(parameter.MA_days) is int
                MA_width = (data_for_baseline.samples_per_day * parameter.MA_days)
            input_width = int(dataUtil.samples_per_day * parameter.input_days)
            label_width = int(dataUtil.samples_per_day * parameter.output_days)
            shift = int(dataUtil.samples_per_day * parameter.shifted_days)
            log.info("\n------Cross-day Prediction------")
            log.info("input days: {}".format(parameter.input_days))
            log.info("shift days: {}".format(parameter.shifted_days))
            log.info("output days: {}".format(parameter.output_days))
            if "MA" in parameter.model_list:
                log.info("MA days: {}".format(parameter.MA_days))
            log.info("samples per day: {}".format(dataUtil.samples_per_day))
            log.info("input width: {}".format(input_width))
            log.info("shift width: {}".format(shift))
            log.info("label width: {}".format(label_width))
            if parameter.is_using_image_data:
                log.info("images width: {}".format(image_input_width))
            if "MA" in parameter.model_list:
                log.info("MA width: {}".format(MA_width))
        # w1 = WindowGenerator(input_width=input_width,
        # 						image_input_width=1,
        # 						label_width=label_width,
        # 						shift = parameter.after_minutes,

        # 						trainImages = dataUtil.trainImages,
        # 						trainData = dataUtil.train_df,
        # 						trainCloud = dataUtil.train_df_cloud,######
        # 						trainAverage = dataUtil.train_df_average,######
        # 						trainY = dataUtil.train_df,

        # 						valImage = dataUtil.valImages,
        # 						valData = dataUtil.val_df,
        # 						valCloud = dataUtil.val_df_cloud,######
        # 						valAverage = dataUtil.val_df_average,######
        # 						valY = dataUtil.val_df,

        # 						testImage = dataUtil.testImages,
        # 						testData = dataUtil.test_df,
        # 						testCloud = dataUtil.test_df_cloud,######
        # 						testAverage = dataUtil.test_df_average,######
        # 						testY = dataUtil.test_df,

        # 						batch_size = parameter.batchsize,
        # 						label_columns = "ShortWaveDown")
        # log.info(w1)	#2D
        # w1.checkWindow()

        w2 = WindowGenerator(input_width=input_width,
                             image_input_width=image_input_width,
                             label_width=label_width,
                             shift=shift,

                             trainImages=dataUtil.trainImages,
                             trainData=dataUtil.train_df[dataUtil.feature_col],
                             trainCloud=dataUtil.train_df_cloud,  ######
                             trainAverage=dataUtil.train_df_average,  ######
                             trainY=dataUtil.train_df[dataUtil.label_col],

                             valImage=dataUtil.valImages,
                             valData=dataUtil.val_df[dataUtil.feature_col],
                             valCloud=dataUtil.val_df_cloud,  ######
                             valAverage=dataUtil.val_df_average,  ######
                             valY=dataUtil.val_df[dataUtil.label_col],

                             testImage=dataUtil.testImages,
                             testData=dataUtil.test_df[dataUtil.feature_col],
                             testCloud=dataUtil.test_df_cloud,  ######
                             testAverage=dataUtil.test_df_average,  ######
                             testY=dataUtil.test_df[dataUtil.label_col],

                             batch_size=parameter.batchsize,
                             label_columns="ShortWaveDown",
                             samples_per_day=dataUtil.samples_per_day,
                             using_timestamp_data=False)
        w_with_timestamp_data = WindowGenerator(input_width=input_width,
                                                image_input_width=image_input_width,
                                                label_width=label_width,
                                                shift=shift,

                                                trainImages=dataUtil.trainImages,
                                                trainData=dataUtil.train_df[dataUtil.feature_col],
                                                trainCloud=dataUtil.train_df_cloud,  ######
                                                trainAverage=dataUtil.train_df_average,  ######
                                                trainY=dataUtil.train_df[dataUtil.label_col],

                                                valImage=dataUtil.valImages,
                                                valData=dataUtil.val_df[dataUtil.feature_col],
                                                valCloud=dataUtil.val_df_cloud,  ######
                                                valAverage=dataUtil.val_df_average,  ######
                                                valY=dataUtil.val_df[dataUtil.label_col],

                                                testImage=dataUtil.testImages,
                                                testData=dataUtil.test_df[dataUtil.feature_col],
                                                testCloud=dataUtil.test_df_cloud,  ######
                                                testAverage=dataUtil.test_df_average,  ######
                                                testY=dataUtil.test_df[dataUtil.label_col],

                                                batch_size=parameter.batchsize,
                                                label_columns="ShortWaveDown",
                                                samples_per_day=dataUtil.samples_per_day,
                                                using_timestamp_data=True)
        # log.info(w2)  # 3D
        dataUtil = data_for_baseline
        if "Persistence" in parameter.model_list:
            w_for_persistance = WindowGenerator(input_width=input_width,
                                                image_input_width=image_input_width,
                                                label_width=label_width,
                                                shift=shift,

                                                trainImages=dataUtil.trainImages,
                                                trainData=dataUtil.train_df[dataUtil.feature_col],
                                                trainCloud=dataUtil.train_df_cloud,  ######
                                                trainAverage=dataUtil.train_df_average,  ######
                                                trainY=dataUtil.train_df[dataUtil.label_col],

                                                valImage=dataUtil.valImages,
                                                valData=dataUtil.val_df[dataUtil.feature_col],
                                                valCloud=dataUtil.val_df_cloud,  ######
                                                valAverage=dataUtil.val_df_average,  ######
                                                valY=dataUtil.val_df[dataUtil.label_col],

                                                testImage=dataUtil.testImages,
                                                testData=dataUtil.test_df[dataUtil.feature_col],
                                                testCloud=dataUtil.test_df_cloud,  ######
                                                testAverage=dataUtil.test_df_average,  ######
                                                testY=dataUtil.test_df[dataUtil.label_col],

                                                batch_size=parameter.batchsize,
                                                label_columns="ShortWaveDown",
                                                samples_per_day=dataUtil.samples_per_day)
        if "MA" in parameter.model_list:
            w_for_MA = WindowGenerator(input_width=MA_width,
                                       image_input_width=image_input_width,
                                       label_width=label_width,
                                       shift=shift,

                                       trainImages=dataUtil.trainImages,
                                       trainData=dataUtil.train_df[dataUtil.feature_col],
                                       trainCloud=dataUtil.train_df_cloud,  ######
                                       trainAverage=dataUtil.train_df_average,  ######
                                       trainY=dataUtil.train_df[dataUtil.label_col],

                                       valImage=dataUtil.valImages,
                                       valData=dataUtil.val_df[dataUtil.feature_col],
                                       valCloud=dataUtil.val_df_cloud,  ######
                                       valAverage=dataUtil.val_df_average,  ######
                                       valY=dataUtil.val_df[dataUtil.label_col],

                                       testImage=dataUtil.testImages,
                                       testData=dataUtil.test_df[dataUtil.feature_col],
                                       testCloud=dataUtil.test_df_cloud,  ######
                                       testAverage=dataUtil.test_df_average,  ######
                                       testY=dataUtil.test_df[dataUtil.label_col],

                                       batch_size=parameter.batchsize,
                                       label_columns="ShortWaveDown",
                                       samples_per_day=dataUtil.samples_per_day)
        # w2.checkWindow()
        # w3 = WindowGenerator(input_width=input_width,
        # 						image_input_width=image_input_width,
        # 						label_width=1,
        # 						shift = parameter.after_minutes,

        # 						trainImages = dataUtil.trainImages,
        # 						trainData = dataUtil.train_df,
        # 						trainCloud = dataUtil.train_df_cloud,######
        # 						trainAverage = dataUtil.train_df_average,######
        # 						trainY = dataUtil.train_df,

        # 						valImage = dataUtil.valImages,
        # 						valData = dataUtil.val_df,
        # 						valCloud = dataUtil.val_df_cloud,######
        # 						valAverage = dataUtil.val_df_average,######
        # 						valY = dataUtil.val_df,

        # 						testImage = dataUtil.testImages,
        # 						testData = dataUtil.test_df,
        # 						testCloud = dataUtil.test_df_cloud,######
        # 						testAverage = dataUtil.test_df_average,######
        # 						testY = dataUtil.test_df,

        # 						batch_size = parameter.batchsize,
        # 						label_columns = "ShortWaveDown")
        # log.info(w3)	#3D
        # w3.checkWindow()

        w = w2
    #############################################################
    log = logging.getLogger(parameter.experient_label)
    # if parameter.dynamic_model == "two":
    #     sep_trainA = w.trainAC(sepMode="cloudA")
    # elif parameter.dynamic_model == "one":
    #     if parameter.addAverage:
    #         sep_trainA = w.train(addcloud=True)
    #     else:
    #         sep_trainA = w.train(addcloud=False)
    # log.info("Dataset shape: {}".format(sep_trainA))


    is_input_continuous_with_output = (shift == 0) and (not parameter.between8_17 or w.is_sampling_within_day)
    metrics_path = "plot/{}/{}".format(parameter.experient_label, "all_metric")

    # test baseline model
    dataUtil = data_for_baseline
    if "Persistence" in parameter.model_list:
        baseline = Persistence(w_for_persistance.is_sampling_within_day,
                               w_for_persistance.samples_per_day,
                               label_width)
        baseline.compile(loss=tf.losses.MeanSquaredError(),
                         metrics=[tf.metrics.MeanAbsoluteError()
                             , tf.metrics.MeanAbsolutePercentageError()
                             , my_metrics.VWMAPE
                             , my_metrics.corr])
        # performance = {}
        modelList = {}
        metricsDict = w_for_persistance.allPlot(model=[baseline],
                                                name="Persistence",
                                                scaler=dataUtil.labelScaler,
                                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["Persistence"] = metricsDict[logM]
        metrics_path = "plot/{}/{}".format(parameter.experient_label, "all_metric")
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "MA" in parameter.model_list:
        movingAverage = MA(MA_width, w_for_MA.is_sampling_within_day, w_for_MA.samples_per_day, label_width)
        movingAverage.compile(loss=tf.losses.MeanSquaredError(),
                              metrics=[tf.metrics.MeanAbsoluteError(),
                                       tf.metrics.MeanAbsolutePercentageError(),
                                       my_metrics.VWMAPE,
                                       my_metrics.corr])
        metricsDict = w_for_MA.allPlot(model=[movingAverage],
                                       name="MA",
                                       scaler=dataUtil.labelScaler,
                                       datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["MA"] = metricsDict[logM]
        metrics_path = "plot/{}/{}".format(parameter.experient_label, "all_metric")
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    # test learning models
    dataUtil = data_with_weather_info
    w = w2
    if "conv3D_c_cnnlstm" in parameter.model_list:
        parameter.squeeze = False
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("training conv3D_c_cnnlstm...")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            image_model = model_conv3D.conv3D(image_input_width, 48, 64, regress=False)  ### #
            # data_model = models.cnnlstm_david(input_width, 1, regress=False)##############
            datamodel_CL = models.cnnlstm_david(input_width, 1, regress=False)
            if parameter.addAverage == True:
                if parameter.dynamic_model == "one" and parameter.addAverage == True:
                    datamodel_average = models.cnnlstm_david(input_width, 2, regress=False)  ###
                else:
                    datamodel_average = models.cnnlstm_david(input_width, 1, regress=False)  ### #
                combinedInput = concatenate([datamodel_CL.output, datamodel_average.output])
                # print("&&&&&&",combinedInput)#
                # z = Dense(label_width)(combinedInput)#
                # print(z)#
                data_model = Model(inputs=[datamodel_CL.input, datamodel_average.input], outputs=combinedInput)  ### #
            else:
                data_model = datamodel_CL
            ####################################################################################
            combinedInput = concatenate([image_model.output, data_model.output])
            x = Dense(1)(combinedInput)  #
            combined2 = Model(inputs=[image_model.input, data_model.input], outputs=x)
            # combined2.summary()
            if parameter.dynamic_model == "two":
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                image_model2 = model_conv3D.conv3D(image_input_width, 48, 64, predict_length=label_width,
                                                   regress=False)  ### #
                # data_model2 = models.cnnlstm_david(input_width, 1, regress=False)############
                datamodel_CL_2 = models.cnnlstm_david(input_width, 1, regress=False)
                if parameter.addAverage == True:
                    datamodel_average_2 = models.cnnlstm_david(input_width, 1, regress=False)  ### #
                    combinedInput_2 = concatenate([datamodel_CL_2.output, datamodel_average_2.output])
                    # print("&&&&&&",combinedInput_2)
                    # z_2 = Dense(label_width)(combinedInput_2)
                    # print(z_2)
                    data_model2 = Model(inputs=[datamodel_CL_2.input, datamodel_average_2.input],
                                        outputs=combinedInput_2)  ### #
                else:
                    data_model2 = datamodel_CL_2
                ####################################################################################
                combinedInput2 = concatenate([image_model2.output, data_model2.output])
                x2 = Dense(1)(combinedInput2)  #
                combined2_2 = Model(inputs=[image_model2.input, data_model2.input], outputs=x2)
                # combined2_2.summary()
                model1, val_performance1, model2, val_performance2 = ModelTrainer_cloud(dataGnerator=w,
                                                                                        model1=combined2,
                                                                                        model2=combined2_2,
                                                                                        generatorMode="combined",
                                                                                        testEpoch=testEpoch,
                                                                                        name="conv3D_c_cnnlstm")
                print(val_performance1, val_performance2)
                if ((best_perform == None) or (best_perform[2] > val_performance1[2])):
                    best_model = model1
                    best_perform = val_performance1
                print(best_perform)
                if ((best_perform2 == None) or (best_perform2[2] > val_performance2[2])):
                    best_model2 = model2
                    best_perform2 = val_performance2
                print(best_perform2)
                log.info("val per1: {}  ///best per1: {}".format(val_performance1, best_perform))
                log.info("val per2: {}  ///best per2: {}".format(val_performance2, best_perform2))
                log.info("a model ok")
            else:
                combined2, combined2_performance = ModelTrainer(dataGnerator=w, model=combined2,
                                                                generatorMode="combined", testEpoch=testEpoch,
                                                                name="conv3D_c_cnnlstm")
                print(combined2_performance)
                if ((best_perform == None) or (best_perform[3] > combined2_performance[3])):
                    best_model = combined2
                    best_perform = combined2_performance
                print(best_perform)
                log.info("a model ok")

        log.info("predicting SolarIrradiation by conv3D_c_cnnlstm...")
        # tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
        # modelList["conv3D_c_cnnlstm"] = [combined2]
        if parameter.dynamic_model == "two":
            metricsDict = w.allPlot(model=[best_model, best_model2],
                                    name="conv3D_c_cnnlstm",
                                    scaler=dataUtil.labelScaler,
                                    datamode="combined")
        elif parameter.dynamic_model == "one":
            metricsDict = w.allPlot(model=[best_model],
                                    name="conv3D_c_cnnlstm",
                                    scaler=dataUtil.labelScaler,
                                    datamode="combined")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["conv3D_c_cnnlstm"] = metricsDict[logM]
        metrics_path = "plot/{}/{}".format(parameter.experient_label, "all_metric")
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "Cnn3dLSTM_c_cnnlstm" in parameter.model_list:
        parameter.squeeze = False
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("training Cnn3dLSTM_c_cnnlstm...")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            image_model = model_cnnLSTM.cnnLSTM(image_input_width, 48, 64, 3, regress=True)
            # data_model = models.cnnlstm_david(input_width, regress=False)##############
            datamodel_CL = models.cnnlstm_david(input_width, 1, regress=False)
            datamodel_average = models.cnnlstm_david(input_width, 1, regress=True)
            combinedInput = concatenate([datamodel_CL.output, datamodel_average.output])
            print("&&&&&&", combinedInput)
            z = Dense(label_width)(combinedInput)
            print(z)
            data_model = Model(inputs=[datamodel_CL.input, datamodel_average.input], outputs=z)
            ####################################################################################
            combinedInput = concatenate([image_model.output, data_model.output])
            x = Dense(label_width)(combinedInput)
            combined4 = Model(inputs=[image_model.input, data_model.input], outputs=x)
            # combined2.summary()
            if parameter.dynamic_model == "two":
                image_model2 = model_cnnLSTM.cnnLSTM(image_input_width, 48, 64, 3, regress=True)
                # data_model2 = models.cnnlstm_david(input_width, regress=False)############
                datamodel_CL_2 = models.cnnlstm_david(input_width, 1, regress=False)
                datamodel_average_2 = models.cnnlstm_david(input_width, 1, regress=True)
                combinedInput_2 = concatenate([datamodel_CL_2.output, datamodel_average_2.output])
                print("&&&&&&", combinedInput_2)
                z_2 = Dense(label_width)(combinedInput_2)
                print(z_2)
                data_model2 = Model(inputs=[datamodel_CL_2.input, datamodel_average_2.input], outputs=z_2)
                ####################################################################################
                combinedInput2 = concatenate([image_model2.output, data_model2.output])
                x2 = Dense(label_width)(combinedInput2)
                combined4_2 = Model(inputs=[image_model2.input, data_model2.input], outputs=x2)
                # combined4.summary()
                model1, val_performance1, model2, val_performance2 = ModelTrainer_cloud(dataGnerator=w,
                                                                                        model1=combined4,
                                                                                        model2=combined4_2,
                                                                                        generatorMode="combined",
                                                                                        testEpoch=testEpoch,
                                                                                        name="Cnn3dLSTM_c_cnnlstm",
                                                                                        sep_trainA=sep_trainA,
                                                                                        sep_trainC=sep_trainC,
                                                                                        sep_valA=sep_valA,
                                                                                        sep_valC=sep_valC)
                print(val_performance1, val_performance2)
                if ((best_perform == None) or (best_perform[2] > val_performance1[2])):
                    best_model = model1
                    best_perform = val_performance1
                print(best_perform)
                if ((best_perform2 == None) or (best_perform2[2] > val_performance2[2])):
                    best_model2 = model2
                    best_perform2 = val_performance2
                print(best_perform2)
                log.info("a model ok")

            else:
                combined4, combined4_performance = ModelTrainer(dataGnerator=w, model=combined4,
                                                                generatorMode="combined", testEpoch=testEpoch)
                print(combined4_performance)
                if ((best_perform == None) or (best_perform[3] > combined4_performance[3])):
                    best_model = combined4
                    best_perform = combined4_performance
                print(best_perform)
                log.info("a model ok")

        log.info("predicting SolarIrradiation by Cnn3dLSTM_c_cnnlstm...")
        # tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
        # modelList["Cnn3dLSTM_c_cnnlstm"] = [combined4]
        if parameter.dynamic_model == "two":
            metricsDict = w.allPlot(model=[best_model, best_model2],
                                    name="Cnn3dLSTM_c_cnnlstm",
                                    scaler=dataUtil.labelScaler,
                                    datamode="combined")
        if parameter.dynamic_model == "one":
            metricsDict = w.allPlot(model=[best_model],
                                    name="Cnn3dLSTM_c_cnnlstm",
                                    scaler=dataUtil.labelScaler,
                                    datamode="combined")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["Cnn3dLSTM_c_cnnlstm"] = metricsDict[logM]

    if "Resnet50_c_cnnlstm" in parameter.model_list:
        parameter.squeeze = False
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("training Resnet50_c_cnnlstm...")
        input_shape = (image_input_width, 48, 64, 3)
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            tf.keras.backend.clear_session()
            inp = Input(shape=input_shape)
            c1 = inp[:, 0, :, :, :]
            c2 = inp[:, 1, :, :, :]
            c3 = inp[:, 2, :, :, :]
            c4 = inp[:, 3, :, :, :]
            c5 = inp[:, 4, :, :, :]
            merged = tf.concat([c1, c2, c3, c4, c5], axis=0)
            resnet1 = ResNet50(include_top=False, weights="imagenet", input_tensor=merged, input_shape=(64, 48, 3))
            for layer in resnet1.layers:
                layer.trainable = False
                layer._name = layer.name + str('_1')
            last = resnet1.output
            x = Flatten()(last)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(5, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1)(x)
            x = tf.reshape(x, [-1, 5])
            resnet2 = ResNet50(include_top=False, weights="imagenet", input_tensor=c5, input_shape=(64, 48, 3))
            for layer in resnet2.layers:
                layer.trainable = False
                layer._name = layer.name + str('_2')
            last = resnet2.output
            y = Flatten()(last)
            y = Dense(256, activation='relu')(y)
            y = Dropout(0.5)(y)
            y = Dense(5, activation='relu')(y)
            xy = concatenate([x, y])
            xy = Dense(1)(xy)
            image_model = Model(inputs=resnet1.input, outputs=xy)
            data_model = models.cnnlstm_david(input_width, 1, regress=False)  #########
            # datamodel_CL = models.cnnlstm_david(input_width, 1, regress=False)
            # datamodel_cloud = models.cnnlstm_david(input_width, 2, regress=True)
            # combinedInput = concatenate([datamodel_CL.output, datamodel_cloud.output])
            # print("&&&&&&",combinedInput)
            # z = Dense(label_width)(combinedInput)
            # print(z)
            # data_model = Model(inputs=[datamodel_CL.input, datamodel_cloud.input], outputs=z)
            ####################################################################
            combinedInput = concatenate([image_model.output, data_model.output])
            z = Dense(label_width)(combinedInput)
            pretrain1 = Model(inputs=[image_model.input, data_model.input], outputs=z)

            if parameter.dynamic_model == "two":
                inp = Input(shape=input_shape)
                c1 = inp[:, 0, :, :, :]
                c2 = inp[:, 1, :, :, :]
                c3 = inp[:, 2, :, :, :]
                c4 = inp[:, 3, :, :, :]
                c5 = inp[:, 4, :, :, :]
                merged = tf.concat([c1, c2, c3, c4, c5], axis=0)
                resnet1 = ResNet50(include_top=False, weights="imagenet", input_tensor=merged, input_shape=(64, 48, 3))
                for layer in resnet1.layers:
                    layer.trainable = False
                    layer._name = layer.name + str('_1')
                last = resnet1.output
                x = Flatten()(last)
                x = Dense(256, activation='relu')(x)
                x = Dropout(0.5)(x)
                x = Dense(5, activation='relu')(x)
                x = Dropout(0.5)(x)
                x = Dense(1)(x)
                x = tf.reshape(x, [-1, 5])
                resnet2 = ResNet50(include_top=False, weights="imagenet", input_tensor=c5, input_shape=(64, 48, 3))
                for layer in resnet2.layers:
                    layer.trainable = False
                    layer._name = layer.name + str('_2')
                last = resnet2.output
                y = Flatten()(last)
                y = Dense(256, activation='relu')(y)
                y = Dropout(0.5)(y)
                y = Dense(5, activation='relu')(y)
                xy = concatenate([x, y])
                xy = Dense(1)(xy)
                image_model2 = Model(inputs=resnet1.input, outputs=xy)
                data_model2 = models.cnnlstm_david(input_width, 1, regress=False)
                combinedInput = concatenate([image_model.output, data_model.output])
                z = Dense(label_width)(combinedInput)
                pretrain1_2 = Model(inputs=[image_model.input, data_model.input], outputs=z)

                model1, val_performance1, model2, val_performance2 = ModelTrainer_cloud(dataGnerator=w,
                                                                                        model1=pretrain1,
                                                                                        model2=pretrain1_2,
                                                                                        generatorMode="combined",
                                                                                        testEpoch=testEpoch,
                                                                                        name="Resnet50_c_cnnlstm")
                print(val_performance1, val_performance2)
                if ((best_perform == None) or (best_perform[3] > val_performance1[3])):
                    best_model = model1
                    best_perform = val_performance1
                print(best_perform)
                if ((best_perform2 == None) or (best_perform2[3] > val_performance2[3])):
                    best_model2 = model2
                    best_perform2 = val_performance2
                print(best_perform2)
                log.info("a model ok")

            else:
                pretrain1, pretrain1_performance = ModelTrainer(dataGnerator=w, model=pretrain1,
                                                                generatorMode="combined", testEpoch=testEpoch)
                print(pretrain1_performance)
                if ((best_perform == None) or (best_perform[3] > pretrain1_performance[3])):
                    best_model = pretrain1
                    best_perform = pretrain1_performance
                print(best_perform)
                log.info("a model ok")

        log.info("predicting SolarIrradiation by Resnet50_c_cnnlstm...")
        # tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
        # modelList["Cnn3dLSTM_c_cnnlstm"] = [combined4]
        if parameter.dynamic_model == "two":
            metricsDict = w.allPlot(model=[best_model, best_model2],
                                    name="Resnet50_c_cnnlstm",
                                    scaler=dataUtil.labelScaler,
                                    datamode="combined")
        if parameter.dynamic_model == "one":
            metricsDict = w.allPlot(model=[best_model],
                                    name="Resnet50_c_cnnlstm",
                                    scaler=dataUtil.labelScaler,
                                    datamode="combined")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["Resnet50_c_cnnlstm"] = metricsDict[logM]

    if "Efficient_c_cnnlstm" in parameter.model_list:
        parameter.squeeze = False
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("training Efficient_c_cnnlstm...")
        input_shape = (input_width, 48, 64, 3)
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            inp = Input(shape=input_shape)
            c1 = inp[:, 0, :, :, :]
            c2 = inp[:, 1, :, :, :]
            c3 = inp[:, 2, :, :, :]
            c4 = inp[:, 3, :, :, :]
            c5 = inp[:, 4, :, :, :]
            merged = tf.concat([c1, c2, c3, c4, c5], axis=0)
            Efficient1 = tf.keras.applications.EfficientNetB7(include_top=False, weights="imagenet",
                                                              input_tensor=merged, input_shape=(64, 48, 3))
            for layer in Efficient1.layers:
                layer.trainable = False
                layer._name = layer.name + str('_1')
            last = Efficient1.output
            x = Flatten()(last)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(5, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1)(x)
            x = tf.reshape(x, [-1, 5])
            Efficient2 = tf.keras.applications.EfficientNetB7(include_top=False, weights="imagenet", input_tensor=c5,
                                                              input_shape=(64, 48, 3))
            for layer in Efficient2.layers:
                layer.trainable = False
                layer._name = layer.name + str('_2')
            last = Efficient2.output
            y = Flatten()(last)
            y = Dense(256, activation='relu')(y)
            y = Dropout(0.5)(y)
            y = Dense(5, activation='relu')(y)
            xy = concatenate([x, y])
            xy = Dense(1)(xy)
            image_model = Model(inputs=Efficient1.input, outputs=xy)
            data_model = models.cnnlstm_david(input_width, regress=False, expand=False)
            combinedInput = concatenate([image_model.output, data_model.output])
            z = Dense(label_width)(combinedInput)
            pretrain2 = Model(inputs=[image_model.input, data_model.input], outputs=z)

            if parameter.dynamic_model == "two":
                inp = Input(shape=input_shape)
                c1 = inp[:, 0, :, :, :]
                c2 = inp[:, 1, :, :, :]
                c3 = inp[:, 2, :, :, :]
                c4 = inp[:, 3, :, :, :]
                c5 = inp[:, 4, :, :, :]
                merged = tf.concat([c1, c2, c3, c4, c5], axis=0)
                Efficient1 = tf.keras.applications.EfficientNetB7(include_top=False, weights="imagenet",
                                                                  input_tensor=merged, input_shape=(64, 48, 3))
                for layer in Efficient1.layers:
                    layer.trainable = False
                    layer._name = layer.name + str('_1')
                last = Efficient1.output
                x = Flatten()(last)
                x = Dense(256, activation='relu')(x)
                x = Dropout(0.5)(x)
                x = Dense(5, activation='relu')(x)
                x = Dropout(0.5)(x)
                x = Dense(1)(x)
                x = tf.reshape(x, [-1, 5])
                Efficient2 = tf.keras.applications.EfficientNetB7(include_top=False, weights="imagenet",
                                                                  input_tensor=merged, input_shape=(64, 48, 3))
                for layer in Efficient2.layers:
                    layer.trainable = False
                    layer._name = layer.name + str('_2')
                last = Efficient2.output
                y = Flatten()(last)
                y = Dense(256, activation='relu')(y)
                y = Dropout(0.5)(y)
                y = Dense(5, activation='relu')(y)
                xy = concatenate([x, y])
                xy = Dense(1)(xy)
                image_model2 = Model(inputs=Efficient1.input, outputs=xy)
                data_model2 = models.cnnlstm_david(input_width, regress=False, expand=False)
                combinedInput = concatenate([image_model.output, data_model.output])
                z = Dense(label_width)(combinedInput)
                pretrain2_2 = Model(inputs=[image_model.input, data_model.input], outputs=z)

                model1, val_performance1, model2, val_performance2 = ModelTrainer_cloud(dataGnerator=w,
                                                                                        model1=pretrain2,
                                                                                        model2=pretrain2_2,
                                                                                        generatorMode="combined",
                                                                                        testEpoch=testEpoch,
                                                                                        name="Efficient_c_cnnlstm")
                print(val_performance1, val_performance2)
                if ((best_perform == None) or (best_perform[3] > val_performance1[3])):
                    best_model = model1
                    best_perform = val_performance1
                print(best_perform)
                if ((best_perform2 == None) or (best_perform2[3] > val_performance2[3])):
                    best_model2 = model2
                    best_perform2 = val_performance2
                print(best_perform2)
                log.info("a model ok")

            else:
                pretrain2, pretrain2_performance = ModelTrainer(dataGnerator=w, model=pretrain2,
                                                                generatorMode="combined", testEpoch=testEpoch)
                print(pretrain2_performance)
                if ((best_perform == None) or (best_perform[3] > pretrain2_performance[3])):
                    best_model = pretrain2
                    best_perform = pretrain2_performance
                print(best_perform)
                log.info("a model ok")

        log.info("predicting SolarIrradiation by Efficient_c_cnnlstm...")
        # tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
        # modelList["Cnn3dLSTM_c_cnnlstm"] = [combined4]
        if parameter.dynamic_model == "two":
            metricsDict = w.allPlot(model=[best_model, best_model2],
                                    name="Efficient_c_cnnlstm",
                                    scaler=dataUtil.labelScaler,
                                    datamode="combined")
        if parameter.dynamic_model == "one":
            metricsDict = w.allPlot(model=[best_model],
                                    name="Efficient_c_cnnlstm",
                                    scaler=dataUtil.labelScaler,
                                    datamode="combined")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["Efficient_c_cnnlstm"] = metricsDict[logM]
    ###########################################################################
    # if "data_cnnlstm" in parameter.model_list:
    #     # sep_trainDataA = w.trainDataAC(sepMode="cloudA")
    #     # sep_trainDataC = w.trainDataAC(sepMode="cloudC")
    #     # sep_valDataA = w.valDataAC(sepMode="cloudA")
    #     # sep_valDataC = w.valDataAC(sepMode="cloudC")
    #     # log.info("Dataset shape: {}".format(sep_trainDataA))
    #     best_perform, best_perform2 = None, None
    #     best_model, best_model2 = None, None
    #     log.info("training datamodel_CL...")
    #     for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
    #         datamodel_CL = models.cnnlstm_david(input_width, len(parameter.features), label_width, regress=False)
    #         if parameter.addAverage == True:
    #             if parameter.dynamic_model == "one" and parameter.addAverage == True:
    #                 datamodel_cloud = models.cnnlstm_david(input_width, 2, label_width, regress=False)
    #             else:
    #                 datamodel_cloud = models.cnnlstm_david(input_width, 1, label_width, regress=False)
    #             combinedInput = concatenate([datamodel_CL.output, datamodel_cloud.output])
    #             print("&&&&&&", combinedInput)
    #             z = Dense(1)(combinedInput)
    #             print(z)
    #             data_model = Model(inputs=[datamodel_CL.input, datamodel_cloud.input], outputs=z)
    #         else:
    #             data_model = datamodel_CL
    #         if parameter.dynamic_model == "two":
    #             datamodel_CL_2 = models.cnnlstm_david(input_width, len(parameter.features), regress=False)
    #             if parameter.addAverage == True:
    #                 datamodel_cloud_2 = models.cnnlstm_david(input_width, 1, regress=True)
    #                 combinedInput_2 = concatenate([datamodel_CL_2.output, datamodel_cloud_2.output])
    #                 print("&&&&&&", combinedInput_2)
    #                 z = Dense(label_width)(combinedInput_2)
    #                 print(z)
    #                 data_model2 = Model(inputs=[datamodel_CL_2.input, datamodel_cloud_2.input], outputs=z)
    #             else:
    #                 data_model2 = datamodel_CL
    #
    #             model1, val_performance1, model2, val_performance2 = ModelTrainer_cloud(dataGnerator=w,
    #                                                                                     model1=data_model,
    #                                                                                     model2=data_model2,
    #                                                                                     generatorMode="data",
    #                                                                                     testEpoch=testEpoch,
    #                                                                                     name="datamodel_CL")
    #             print(val_performance1, val_performance2)
    #             if ((best_perform == None) or (best_perform[2] > val_performance1[2])):
    #                 best_model = model1
    #                 best_perform = val_performance1
    #             print(best_perform)
    #             if ((best_perform2 == None) or (best_perform2[2] > val_performance2[2])):
    #                 best_model2 = model2
    #                 best_perform2 = val_performance2
    #             print(best_perform2)
    #             log.info("a model ok")
    #         else:
    #             datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=data_model,
    #                                                                   generatorMode="data", testEpoch=testEpoch,
    #                                                                   name="datamodel_CL")
    #             print(datamodel_CL_performance)
    #             if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
    #                 best_model = datamodel_CL
    #                 best_perform = datamodel_CL_performance
    #             print(best_perform)
    #             log.info("a model ok")
    #
    #     log.info("predicting SolarIrradiation by datamodel_CL...")
    #     # tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    #     # modelList["conv3D_c_cnnlstm"] = [combined2]
    #     if parameter.dynamic_model == "two":
    #         metricsDict = w.allPlot(model=[best_model, best_model2],
    #                                 name="datamodel_CL",
    #                                 scaler=dataUtil.labelScaler,
    #
    #                                 datamode="data")
    #     elif parameter.dynamic_model == "one":
    #         metricsDict = w.allPlot(model=[best_model],
    #                                 name="datamodel_CL",
    #                                 scaler=dataUtil.labelScaler,
    #
    #                                 datamode="data")
    #     for logM in metricsDict:
    #         if modelMetricsRecorder.get(logM) is None:
    #             modelMetricsRecorder[logM] = {}
    #         modelMetricsRecorder[logM]["datamodel_CL"] = metricsDict[logM]
    #     pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))
    # if "simple_transformer" in parameter.model_list:
    #     best_perform, best_perform2 = None, None
    #     best_model, best_model2 = None, None
    #     log.info("training simple_transformer model...")
    #     for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
    #         if parameter.addAverage == True:
    #             if parameter.dynamic_model == "one" and parameter.addAverage == True:
    #                 data_model = models.simple_transformer(8, 4, 3, len(parameter.features), len(parameter.target), 2,
    #                                                        input_width, label_width)
    #             else:
    #                 data_model = models.simple_transformer(8, 4, 3, len(parameter.features), len(parameter.target), 1,
    #                                                        input_width, label_width)
    #         else:
    #             data_model = models.simple_transformer(8, 4, 3, len(parameter.features), len(parameter.target), None,
    #                                                    input_width, label_width)
    #         if parameter.dynamic_model == "two":
    #             if parameter.addAverage == True:
    #                 data_model2 = models.simple_transformer(8, 4, 3, len(parameter.features), len(parameter.target), 1,
    #                                                         input_width, label_width)
    #             else:
    #                 data_model2 = models.simple_transformer(8, 4, 3, len(parameter.features), len(parameter.target),
    #                                                         None, input_width, label_width)
    #
    #             model1, val_performance1, model2, val_performance2 = ModelTrainer_cloud(dataGnerator=w,
    #                                                                                     model1=data_model,
    #                                                                                     model2=data_model2,
    #                                                                                     generatorMode="data",
    #                                                                                     testEpoch=testEpoch,
    #                                                                                     name="simple_transformer")
    #             print(val_performance1, val_performance2)
    #             if ((best_perform == None) or (best_perform[2] > val_performance1[2])):
    #                 best_model = model1
    #                 best_perform = val_performance1
    #             print(best_perform)
    #             if ((best_perform2 == None) or (best_perform2[2] > val_performance2[2])):
    #                 best_model2 = model2
    #                 best_perform2 = val_performance2
    #             print(best_perform2)
    #             log.info("a model ok")
    #         else:
    #             datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=data_model,
    #                                                                   generatorMode="data", testEpoch=testEpoch,
    #                                                                   name="simple_transformer")
    #             print(datamodel_CL_performance)
    #             if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
    #                 best_model = datamodel_CL
    #                 best_perform = datamodel_CL_performance
    #             print(best_perform)
    #             log.info("a model ok")
    #
    #     log.info("predicting SolarIrradiation by simple_transformer...")
    #     # tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    #     # modelList["conv3D_c_cnnlstm"] = [combined2]
    #     if parameter.dynamic_model == "two":
    #         metricsDict = w.allPlot(model=[best_model, best_model2],
    #                                 name="simple_transformer",
    #                                 scaler=dataUtil.labelScaler,
    #
    #                                 datamode="data")
    #     elif parameter.dynamic_model == "one":
    #         metricsDict = w.allPlot(model=[best_model],
    #                                 name="simple_transformer",
    #                                 scaler=dataUtil.labelScaler,
    #
    #                                 datamode="data")
    #         try:
    #             os.mkdir(Path("model/{}".format(parameter.experient_label)))
    #         except:
    #             print("model path exist")
    #         best_model.save("model/{}/{}".format(parameter.experient_label, "model"))
    #     for logM in metricsDict:
    #         if modelMetricsRecorder.get(logM) is None:
    #             modelMetricsRecorder[logM] = {}
    #         modelMetricsRecorder[logM]["simple_transformer"] = metricsDict[logM]
    #     pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    # pure numerical models
    if "convGRU" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("convGRU")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=input_width,
                                          in_dim=len(parameter.features),
                                          out_seq_len=label_width, out_dim=len(parameter.target),
                                          units=model_convGRU.Config.gru_units,
                                          filters=model_convGRU.Config.embedding_filters,
                                          gen_mode='unistep',
                                          is_seq_continuous=is_input_continuous_with_output,
                                          rate=model_convGRU.Config.dropout_rate)
            if not w.is_sampling_within_day and parameter.between8_17:
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))),
                                             SplitInputByDay(n_days=parameter.input_days, n_samples=w.samples_per_day),
                                             MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                                   filter_size=preprocess_utils.Config.kernel_size,
                                                                   n_days=parameter.input_days,
                                                                   n_samples=w.samples_per_day),
                                             model])
            else:
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))), model])

            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="convGRU")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by convGRU...")

        metricsDict = w.allPlot(model=[best_model],
                                name="convGRU",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["convGRU"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "transformer" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("training transformer model...")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            if w.is_sampling_within_day:
                token_len = input_width
            else:
                token_len = (min(input_width, label_width) // w.samples_per_day // 2 + 1) * w.samples_per_day
            if not w.is_sampling_within_day and parameter.between8_17:
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))),
                                             SplitInputByDay(n_days=parameter.input_days, n_samples=w.samples_per_day),
                                             MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                                   filter_size=preprocess_utils.Config.kernel_size,
                                                                   n_days=parameter.input_days,
                                                                   n_samples=w.samples_per_day),
                                             model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                                           d_model=model_transformer.Config.d_model,
                                                                           num_heads=model_transformer.Config.n_heads,
                                                                           dff=model_transformer.Config.dff,
                                                                           src_seq_len=w.samples_per_day,
                                                                           tar_seq_len=label_width,
                                                                           src_dim=preprocess_utils.Config.filters,
                                                                           tar_dim=len(parameter.target),
                                                                           rate=model_transformer.Config.dropout_rate,
                                                                           gen_mode="unistep",
                                                                           is_seq_continuous=is_input_continuous_with_output,
                                                                           is_pooling=False,
                                                                           token_len=0)
                                             ])
            else:
                model = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                      d_model=model_transformer.Config.d_model,
                                                      num_heads=model_transformer.Config.n_heads,
                                                      dff=model_transformer.Config.dff,
                                                      src_seq_len=input_width,
                                                      tar_seq_len=label_width, src_dim=len(parameter.features),
                                                      tar_dim=len(parameter.target),
                                                      rate=model_transformer.Config.dropout_rate,
                                                      gen_mode="unistep",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False,
                                                      token_len=token_len)
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))), model])

            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="transformer")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by transformer...")

        metricsDict = w.allPlot(model=[best_model],
                                name="transformer",
                                scaler=dataUtil.labelScaler,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["transformer"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "convGRU_w_LR" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("convGRU_w_LR")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=input_width,
                                          in_dim=len(parameter.features),
                                          out_seq_len=label_width, out_dim=len(parameter.target),
                                          units=model_convGRU.Config.gru_units,
                                          filters=model_convGRU.Config.embedding_filters,
                                          gen_mode='unistep',
                                          is_seq_continuous=is_input_continuous_with_output,
                                          rate=model_convGRU.Config.dropout_rate)
            if not w.is_sampling_within_day and parameter.between8_17:
                inputs = tf.keras.Input(shape=(input_width, len(parameter.features)))
                linear = model_AR.TemporalChannelIndependentLR(model_AR.Config.order, label_width,
                                                               len(parameter.features))(inputs)
                embedding = SplitInputByDay(n_days=parameter.input_days, n_samples=w.samples_per_day)(inputs)
                embedding = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                  filter_size=preprocess_utils.Config.kernel_size,
                                                  n_days=parameter.input_days,
                                                  n_samples=w.samples_per_day)(embedding)
                nonlinear = model(embedding)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
                model = tf.keras.Model(inputs=inputs, outputs=outputs, name="convGRU_w_LR")
            else:
                inputs = tf.keras.Input(shape=(input_width, len(parameter.features)))
                linear = model_AR.TemporalChannelIndependentLR(model_AR.Config.order, label_width,
                                                               len(parameter.features))(
                    inputs)
                nonlinear = model(inputs)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
                model = tf.keras.Model(inputs=inputs, outputs=outputs, name="convGRU_w_LR")
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="convGRU_w_LR")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by convGRU_w_LR...")

        metricsDict = w.allPlot(model=[best_model],
                                name="convGRU_w_LR",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["convGRU_w_LR"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "transformer_w_LR" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("training transformer_w_LR model...")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            if w.is_sampling_within_day:
                token_len = input_width
            else:
                token_len = (min(input_width, label_width) // w.samples_per_day // 2 + 1) * w.samples_per_day
            if not w.is_sampling_within_day and parameter.between8_17:
                inputs = tf.keras.Input(shape=(input_width, len(parameter.features)))
                linear = model_AR.TemporalChannelIndependentLR(model_AR.Config.order, label_width,
                                                               len(parameter.features))(
                    inputs)
                embedding = SplitInputByDay(n_days=parameter.input_days, n_samples=w.samples_per_day)(inputs)
                embedding = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                  filter_size=preprocess_utils.Config.kernel_size,
                                                  n_days=parameter.input_days,
                                                  n_samples=w.samples_per_day)(embedding)
                nonlinear = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                          d_model=model_transformer.Config.d_model,
                                                          num_heads=model_transformer.Config.n_heads,
                                                          dff=model_transformer.Config.dff,
                                                          src_seq_len=w.samples_per_day,
                                                          tar_seq_len=label_width,
                                                          src_dim=preprocess_utils.Config.filters,
                                                          tar_dim=len(parameter.target),
                                                          rate=model_transformer.Config.dropout_rate,
                                                          gen_mode="unistep",
                                                          is_seq_continuous=is_input_continuous_with_output,
                                                          is_pooling=False,
                                                          token_len=0)(embedding)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
                model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_w_LR")
            else:
                inputs = tf.keras.Input(shape=(input_width, len(parameter.features)))
                linear = model_AR.TemporalChannelIndependentLR(model_AR.Config.order, label_width,
                                                               len(parameter.features))(
                    inputs)
                nonlinear = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                          d_model=model_transformer.Config.d_model,
                                                          num_heads=model_transformer.Config.n_heads,
                                                          dff=model_transformer.Config.dff,
                                                          src_seq_len=input_width,
                                                          tar_seq_len=label_width, src_dim=len(parameter.features),
                                                          tar_dim=len(parameter.target),
                                                          rate=model_transformer.Config.dropout_rate,
                                                          gen_mode="unistep",
                                                          is_seq_continuous=is_input_continuous_with_output,
                                                          is_pooling=False,
                                                          token_len=token_len)(inputs)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
                model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_w_LR")

            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="transformer_w_LR")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by transformer_w_LR...")

        metricsDict = w.allPlot(model=[best_model],
                                name="transformer_w_LR",
                                scaler=dataUtil.labelScaler,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["transformer_w_LR"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "convGRU_w_mlp_decoder" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("convGRU_w_mlp_decoder")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=input_width,
                                          in_dim=len(parameter.features),
                                          out_seq_len=label_width, out_dim=len(parameter.target),
                                          units=model_convGRU.Config.gru_units,
                                          filters=model_convGRU.Config.embedding_filters,
                                          gen_mode='mlp',
                                          is_seq_continuous=is_input_continuous_with_output,
                                          rate=model_convGRU.Config.dropout_rate)
            if not w.is_sampling_within_day and parameter.between8_17:
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))),
                                             SplitInputByDay(n_days=parameter.input_days, n_samples=w.samples_per_day),
                                             MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                                   filter_size=preprocess_utils.Config.kernel_size,
                                                                   n_days=parameter.input_days,
                                                                   n_samples=w.samples_per_day),
                                             model])
            else:
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))), model])

            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="convGRU_w_mlp_decoder")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by convGRU_w_mlp_decoder...")

        metricsDict = w.allPlot(model=[best_model],
                                name="convGRU_w_mlp_decoder",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["convGRU_w_mlp_decoder"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "transformer_w_mlp_decoder" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("training transformer_w_mlp_decoder model...")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            if not w.is_sampling_within_day and parameter.between8_17:
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))),
                                             SplitInputByDay(n_days=parameter.input_days, n_samples=w.samples_per_day),
                                             MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                                   filter_size=preprocess_utils.Config.kernel_size,
                                                                   n_days=parameter.input_days,
                                                                   n_samples=w.samples_per_day),
                                             model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                                           d_model=model_transformer.Config.d_model,
                                                                           num_heads=model_transformer.Config.n_heads,
                                                                           dff=model_transformer.Config.dff,
                                                                           src_seq_len=w.samples_per_day,
                                                                           tar_seq_len=label_width,
                                                                           src_dim=preprocess_utils.Config.filters,
                                                                           tar_dim=len(parameter.target),
                                                                           rate=model_transformer.Config.dropout_rate,
                                                                           gen_mode="mlp",
                                                                           is_seq_continuous=is_input_continuous_with_output,
                                                                           is_pooling=False)
                                             ])
            else:
                model = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                      d_model=model_transformer.Config.d_model,
                                                      num_heads=model_transformer.Config.n_heads,
                                                      dff=model_transformer.Config.dff,
                                                      src_seq_len=input_width,
                                                      tar_seq_len=label_width, src_dim=len(parameter.features),
                                                      tar_dim=len(parameter.target),
                                                      rate=model_transformer.Config.dropout_rate,
                                                      gen_mode="mlp",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False)
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))), model])

            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="transformer_w_mlp_decoder")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by transformer_w_mlp_decoder...")

        metricsDict = w.allPlot(model=[best_model],
                                name="transformer_w_mlp_decoder",
                                scaler=dataUtil.labelScaler,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["transformer_w_mlp_decoder"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "auto_convGRU" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("auto_convGRU")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=input_width,
                                          in_dim=len(parameter.features),
                                          out_seq_len=label_width, out_dim=len(parameter.target),
                                          units=model_convGRU.Config.gru_units,
                                          filters=model_convGRU.Config.embedding_filters,
                                          gen_mode='auto',
                                          is_seq_continuous=is_input_continuous_with_output,
                                          rate=model_convGRU.Config.dropout_rate)
            if not w.is_sampling_within_day and parameter.between8_17:
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))),
                                             SplitInputByDay(n_days=parameter.input_days, n_samples=w.samples_per_day),
                                             MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                                   filter_size=preprocess_utils.Config.kernel_size,
                                                                   n_days=parameter.input_days,
                                                                   n_samples=w.samples_per_day),
                                             model])
            else:
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))), model])
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="auto_convGRU")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by auto_convGRU...")

        metricsDict = w.allPlot(model=[best_model],
                                name="auto_convGRU",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["auto_convGRU"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "auto_transformer" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("auto_transformer")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            if not w.is_sampling_within_day and parameter.between8_17:
                n_days = input_width // w.samples_per_day
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))),
                                             SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day),
                                             MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                                   filter_size=preprocess_utils.Config.kernel_size,
                                                                   n_days=n_days,
                                                                   n_samples=w.samples_per_day),
                                             model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                                           d_model=model_transformer.Config.d_model,
                                                                           num_heads=model_transformer.Config.n_heads,
                                                                           dff=model_transformer.Config.dff,
                                                                           src_seq_len=w.samples_per_day,
                                                                           tar_seq_len=label_width,
                                                                           src_dim=preprocess_utils.Config.filters,
                                                                           tar_dim=len(parameter.target),
                                                                           rate=model_transformer.Config.dropout_rate,
                                                                           gen_mode="auto",
                                                                           is_seq_continuous=is_input_continuous_with_output,
                                                                           is_pooling=False)
                                             ])
            else:
                model = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                      d_model=model_transformer.Config.d_model,
                                                      num_heads=model_transformer.Config.n_heads,
                                                      dff=model_transformer.Config.dff,
                                                      src_seq_len=input_width,
                                                      tar_seq_len=label_width, src_dim=len(parameter.features),
                                                      tar_dim=len(parameter.target),
                                                      rate=model_transformer.Config.dropout_rate,
                                                      gen_mode="auto",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False)
                model = tf.keras.Sequential([tf.keras.Input(shape=(input_width, len(parameter.features))), model])

            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="auto_transformer")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by auto_transformer...")

        metricsDict = w.allPlot(model=[best_model],
                                name="auto_transformer",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["auto_transformer"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "LSTNet" in parameter.model_list:
        assert label_width == 1
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("LSTNet")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            args_dict = GetArgumentsDict()
            init = LSTNetInit(args_dict, True)
            init.window = input_width
            init.skip = w.samples_per_day if w.samples_per_day < input_width else input_width
            init.highway = w.samples_per_day if w.samples_per_day < input_width else input_width
            model = LSTNetModel(init, (None, input_width, len(parameter.features)))
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="LSTNet")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by LSTNet...")

        metricsDict = w.allPlot(model=[best_model],
                                name="LSTNet",
                                scaler=dataUtil.labelScaler,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["LSTNet"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    # intergrated with timestamp data
    dataUtil = data_with_weather_info
    w = w_with_timestamp_data
    if "convGRU_w_timestamps" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("convGRU_w_timestamps")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            input_scalar = Input(shape=(input_width, len(parameter.features)))
            input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
            embedding = time_embedding.TimeEmbedding(output_dims=model_convGRU.Config.embedding_filters,
                                                     input_len=input_width,
                                                     shift_len=shift,
                                                     label_len=label_width)(input_time)
            if not w.is_sampling_within_day and parameter.between8_17:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    embedding[0])
                input_time_embedded = MultipleDaysConvEmbed(filters=model_convGRU.Config.embedding_filters,
                                                            filter_size=preprocess_utils.Config.kernel_size,
                                                            n_days=n_days,
                                                            n_samples=w.samples_per_day)(input_time_embedded)
                model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=w.samples_per_day,
                                              in_dim=len(parameter.features),
                                              out_seq_len=label_width, out_dim=len(parameter.target),
                                              units=model_convGRU.Config.gru_units,
                                              filters=model_convGRU.Config.embedding_filters,
                                              gen_mode='unistep',
                                              is_seq_continuous=is_input_continuous_with_output,
                                              rate=model_convGRU.Config.dropout_rate)
                model = model(scalar_embedded, time_embedding_tuple=(input_time_embedded, embedding[1], embedding[2]))
            else:
                model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=input_width,
                                              in_dim=len(parameter.features),
                                              out_seq_len=label_width, out_dim=len(parameter.target),
                                              units=model_convGRU.Config.gru_units,
                                              filters=model_convGRU.Config.embedding_filters,
                                              gen_mode='unistep',
                                              is_seq_continuous=is_input_continuous_with_output,
                                              rate=model_convGRU.Config.dropout_rate)
                model = model(input_scalar, time_embedding_tuple=embedding)
            model = Model(inputs=[input_scalar, input_time], outputs=model)
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="convGRU_w_timestamps")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by convGRU_w_timestamps...")

        metricsDict = w.allPlot(model=[best_model],
                                name="convGRU_w_timestamps",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["convGRU_w_timestamps"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "transformer_w_timestamps" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("transformer_w_timestamps")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            if w.is_sampling_within_day:
                token_len = input_width
            else:
                token_len = (min(input_width, label_width) // w.samples_per_day // 2 + 1) * w.samples_per_day
            input_scalar = Input(shape=(input_width, len(parameter.features)))
            input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
            embedding = time_embedding.TimeEmbedding(output_dims=model_transformer.Config.d_model,
                                                     input_len=input_width,
                                                     shift_len=shift,
                                                     label_len=label_width)(input_time)
            if not w.is_sampling_within_day and parameter.between8_17:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    embedding[0])
                input_time_embedded = MultipleDaysConvEmbed(filters=model_transformer.Config.d_model,
                                                            filter_size=preprocess_utils.Config.kernel_size,
                                                            n_days=n_days,
                                                            n_samples=w.samples_per_day)(input_time_embedded)
                model = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                      d_model=model_transformer.Config.d_model,
                                                      num_heads=model_transformer.Config.n_heads,
                                                      dff=model_transformer.Config.dff,
                                                      src_seq_len=w.samples_per_day,
                                                      tar_seq_len=label_width,
                                                      src_dim=preprocess_utils.Config.filters,
                                                      tar_dim=len(parameter.target),
                                                      rate=model_transformer.Config.dropout_rate,
                                                      gen_mode="unistep",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False, token_len=0)
                model = model(scalar_embedded, time_embedding_tuple=(input_time_embedded, embedding[1], embedding[2]))
            else:
                model = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                      d_model=model_transformer.Config.d_model,
                                                      num_heads=model_transformer.Config.n_heads,
                                                      dff=model_transformer.Config.dff,
                                                      src_seq_len=input_width,
                                                      tar_seq_len=label_width, src_dim=len(parameter.features),
                                                      tar_dim=len(parameter.target),
                                                      rate=model_transformer.Config.dropout_rate,
                                                      gen_mode="unistep",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False, token_len=token_len)
                model = model(input_scalar, time_embedding_tuple=embedding)
            model = Model(inputs=[input_scalar, input_time], outputs=model)

            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="transformer_w_timestamps")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by transformer_w_timestamps...")
        metricsDict = w.allPlot(model=[best_model],
                                name="transformer_w_timestamps",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["transformer_w_timestamps"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "convGRU_w_LR_timestamps" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("convGRU_w_LR_timestamps")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            input_scalar = Input(shape=(input_width, len(parameter.features)))
            linear = model_AR.TemporalChannelIndependentLR(model_AR.Config.order, label_width,
                                                           len(parameter.features))(input_scalar)
            input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
            embedding = time_embedding.TimeEmbedding(output_dims=model_convGRU.Config.embedding_filters,
                                                     input_len=input_width,
                                                     shift_len=shift,
                                                     label_len=label_width)(input_time)
            if not w.is_sampling_within_day and parameter.between8_17:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    embedding[0])
                input_time_embedded = MultipleDaysConvEmbed(filters=model_convGRU.Config.embedding_filters,
                                                            filter_size=preprocess_utils.Config.kernel_size,
                                                            n_days=n_days,
                                                            n_samples=w.samples_per_day)(input_time_embedded)
                model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=w.samples_per_day,
                                              in_dim=len(parameter.features),
                                              out_seq_len=label_width, out_dim=len(parameter.target),
                                              units=model_convGRU.Config.gru_units,
                                              filters=model_convGRU.Config.embedding_filters,
                                              gen_mode='unistep',
                                              is_seq_continuous=is_input_continuous_with_output,
                                              rate=model_convGRU.Config.dropout_rate)
                nonlinear = model(scalar_embedded,
                                  time_embedding_tuple=(input_time_embedded, embedding[1], embedding[2]))
            else:
                model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=input_width,
                                              in_dim=len(parameter.features),
                                              out_seq_len=label_width, out_dim=len(parameter.target),
                                              units=model_convGRU.Config.gru_units,
                                              filters=model_convGRU.Config.embedding_filters,
                                              gen_mode='unistep',
                                              is_seq_continuous=is_input_continuous_with_output,
                                              rate=model_convGRU.Config.dropout_rate)
                nonlinear = model(input_scalar, time_embedding_tuple=embedding)
            outputs = tf.keras.layers.Add()([linear, nonlinear])
            model = Model(inputs=[input_scalar, input_time], outputs=outputs)
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="convGRU_w_LR_timestamps")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by convGRU_w_LR_timestamps...")

        metricsDict = w.allPlot(model=[best_model],
                                name="convGRU_w_LR_timestamps",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["convGRU_w_LR_timestamps"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "stationary_convGRU_w_LR_timestamps" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("stationary_convGRU_w_LR_timestamps")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            input_scalar = Input(shape=(input_width, len(parameter.features)))
            linear = model_AR.TemporalChannelIndependentLR(model_AR.Config.order, label_width,
                                                           len(parameter.features))(input_scalar)
            input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
            embedding = time_embedding.TimeEmbedding(output_dims=model_convGRU.Config.embedding_filters,
                                                     input_len=input_width,
                                                     shift_len=shift,
                                                     label_len=label_width)(input_time)
            if not w.is_sampling_within_day and parameter.between8_17:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    embedding[0])
                input_time_embedded = MultipleDaysConvEmbed(filters=model_convGRU.Config.embedding_filters,
                                                            filter_size=preprocess_utils.Config.kernel_size,
                                                            n_days=n_days,
                                                            n_samples=w.samples_per_day)(input_time_embedded)
                model = model_convGRU.StationaryConvGRU(num_layers=model_convGRU.Config.layers,
                                                        in_seq_len=w.samples_per_day,
                                                        in_dim=len(parameter.features),
                                                        out_seq_len=label_width, out_dim=len(parameter.target),
                                                        units=model_convGRU.Config.gru_units,
                                                        filters=model_convGRU.Config.embedding_filters,
                                                        gen_mode='unistep',
                                                        is_seq_continuous=is_input_continuous_with_output,
                                                        rate=model_convGRU.Config.dropout_rate,
                                                        avg_window=series_decomposition.Config.window_size)
                nonlinear = model(scalar_embedded,
                                  time_embedding_tuple=(input_time_embedded, embedding[1], embedding[2]))
            else:
                model = model_convGRU.StationaryConvGRU(num_layers=model_convGRU.Config.layers,
                                                        in_seq_len=input_width,
                                                        in_dim=len(parameter.features),
                                                        out_seq_len=label_width, out_dim=len(parameter.target),
                                                        units=model_convGRU.Config.gru_units,
                                                        filters=model_convGRU.Config.embedding_filters,
                                                        gen_mode='unistep',
                                                        is_seq_continuous=is_input_continuous_with_output,
                                                        rate=model_convGRU.Config.dropout_rate,
                                                        avg_window=series_decomposition.Config.window_size)
                nonlinear = model(input_scalar, time_embedding_tuple=embedding)
            outputs = tf.keras.layers.Add()([linear, nonlinear])
            model = Model(inputs=[input_scalar, input_time], outputs=outputs)
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="stationary_convGRU_w_LR_timestamps")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by stationary_convGRU_w_LR_timestamps...")

        metricsDict = w.allPlot(model=[best_model],
                                name="stationary_convGRU_w_LR_timestamps",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["stationary_convGRU_w_LR_timestamps"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "transformer_w_LR_timestamps" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("transformer_w_LR_timestamps")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            if w.is_sampling_within_day:
                token_len = input_width
            else:
                token_len = (min(input_width, label_width) // w.samples_per_day // 2 + 1) * w.samples_per_day
            input_scalar = Input(shape=(input_width, len(parameter.features)))
            linear = model_AR.TemporalChannelIndependentLR(model_AR.Config.order, label_width,
                                                           len(parameter.features))(input_scalar)
            input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
            embedding = time_embedding.TimeEmbedding(output_dims=model_transformer.Config.d_model,
                                                     input_len=input_width,
                                                     shift_len=shift,
                                                     label_len=label_width)(input_time)
            if not w.is_sampling_within_day and parameter.between8_17:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    embedding[0])
                input_time_embedded = MultipleDaysConvEmbed(filters=model_transformer.Config.d_model,
                                                            filter_size=preprocess_utils.Config.kernel_size,
                                                            n_days=n_days,
                                                            n_samples=w.samples_per_day)(input_time_embedded)
                model = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                      d_model=model_transformer.Config.d_model,
                                                      num_heads=model_transformer.Config.n_heads,
                                                      dff=model_transformer.Config.dff,
                                                      src_seq_len=w.samples_per_day,
                                                      tar_seq_len=label_width,
                                                      src_dim=preprocess_utils.Config.filters,
                                                      tar_dim=len(parameter.target),
                                                      rate=model_transformer.Config.dropout_rate,
                                                      gen_mode="unistep",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False, token_len=0)
                nonlinear = model(scalar_embedded,
                                  time_embedding_tuple=(input_time_embedded, embedding[1], embedding[2]))
            else:
                model = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                      d_model=model_transformer.Config.d_model,
                                                      num_heads=model_transformer.Config.n_heads,
                                                      dff=model_transformer.Config.dff,
                                                      src_seq_len=input_width,
                                                      tar_seq_len=label_width, src_dim=len(parameter.features),
                                                      tar_dim=len(parameter.target),
                                                      rate=model_transformer.Config.dropout_rate,
                                                      gen_mode="unistep",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False, token_len=token_len)
                nonlinear = model(input_scalar, time_embedding_tuple=embedding)
            outputs = tf.keras.layers.Add()([linear, nonlinear])
            model = Model(inputs=[input_scalar, input_time], outputs=outputs)

            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="transformer_w_LR_timestamps")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by transformer_w_LR_timestamps...")
        metricsDict = w.allPlot(model=[best_model],
                                name="transformer_w_LR_timestamps",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["transformer_w_LR_timestamps"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "stationary_transformer_w_LR_timestamps" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("stationary_transformer_w_LR_timestamps")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            if w.is_sampling_within_day:
                token_len = input_width
            else:
                token_len = (min(input_width, label_width) // w.samples_per_day // 2 + 1) * w.samples_per_day
            input_scalar = Input(shape=(input_width, len(parameter.features)))
            linear = model_AR.TemporalChannelIndependentLR(model_AR.Config.order, label_width,
                                                           len(parameter.features))(input_scalar)
            input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
            embedding = time_embedding.TimeEmbedding(output_dims=model_transformer.Config.d_model,
                                                     input_len=input_width,
                                                     shift_len=shift,
                                                     label_len=label_width)(input_time)
            if not w.is_sampling_within_day and parameter.between8_17:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    embedding[0])
                input_time_embedded = MultipleDaysConvEmbed(filters=model_transformer.Config.d_model,
                                                            filter_size=preprocess_utils.Config.kernel_size,
                                                            n_days=n_days,
                                                            n_samples=w.samples_per_day)(input_time_embedded)
                model = model_transformer.StationaryTransformer(num_layers=model_transformer.Config.layers,
                                                                d_model=model_transformer.Config.d_model,
                                                                num_heads=model_transformer.Config.n_heads,
                                                                dff=model_transformer.Config.dff,
                                                                src_seq_len=w.samples_per_day,
                                                                tar_seq_len=label_width,
                                                                src_dim=preprocess_utils.Config.filters,
                                                                tar_dim=len(parameter.target),
                                                                rate=model_transformer.Config.dropout_rate,
                                                                gen_mode="unistep",
                                                                is_seq_continuous=is_input_continuous_with_output,
                                                                is_pooling=False, token_len=0,
                                                                avg_window=series_decomposition.Config.window_size)
                nonlinear = model(scalar_embedded,
                                  time_embedding_tuple=(input_time_embedded, embedding[1], embedding[2]))
            else:
                model = model_transformer.StationaryTransformer(num_layers=model_transformer.Config.layers,
                                                                d_model=model_transformer.Config.d_model,
                                                                num_heads=model_transformer.Config.n_heads,
                                                                dff=model_transformer.Config.dff,
                                                                src_seq_len=input_width,
                                                                tar_seq_len=label_width,
                                                                src_dim=len(parameter.features),
                                                                tar_dim=len(parameter.target),
                                                                rate=model_transformer.Config.dropout_rate,
                                                                gen_mode="unistep",
                                                                is_seq_continuous=is_input_continuous_with_output,
                                                                is_pooling=False, token_len=token_len,
                                                                avg_window=series_decomposition.Config.window_size)
                nonlinear = model(input_scalar, time_embedding_tuple=embedding)
            outputs = tf.keras.layers.Add()([linear, nonlinear])
            model = Model(inputs=[input_scalar, input_time], outputs=outputs)

            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="stationary_transformer_w_LR_timestamps")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by stationary_transformer_w_LR_timestamps...")
        metricsDict = w.allPlot(model=[best_model],
                                name="stationary_transformer_w_LR_timestamps",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["stationary_transformer_w_LR_timestamps"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    metrics_path = "plot/{}/{}".format(parameter.experient_label, "all_metric")
    pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    return modelMetricsRecorder


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(run_eagerly=True)
    # tf.data.experimental.enable_debug_mode()
    result = run()
