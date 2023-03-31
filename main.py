# python main.py -d  ../skyImage
# import the necessary packages
import datetime

import tensorflow as tf
from pyimagesearch import datasets, model_AR, time_embedding, time_embedding_factory, bypass_factory, datautil
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
    using_timestamp_data = time_embedding_factory.TEFac.get_te_mode(parameter.time_embedding) is not None
    if generatorMode == "combined" or generatorMode == "data":
        history = model.fit(
            dataGnerator.train(parameter.sample_rate, addcloud=parameter.addAverage,
                               using_timestamp_data=using_timestamp_data,
                               is_shuffle=parameter.is_using_shuffle),
            validation_data=dataGnerator.val(parameter.sample_rate, addcloud=parameter.addAverage,
                                             using_timestamp_data=using_timestamp_data,
                                             is_shuffle=parameter.is_using_shuffle),
            epochs=testEpoch, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
        all_pred, all_y = dataGnerator.plotPredictUnit(model, dataGnerator.val(parameter.sample_rate,
                                                                               addcloud=parameter.addAverage,
                                                                               using_timestamp_data=using_timestamp_data,
                                                                               is_shuffle=parameter.is_using_shuffle),
                                                       datamode=generatorMode)

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
    ap.add_argument("-n", "--experiment_label", type=str, required=True, default=parameter.experiment_label,
                    help="experiment_label")
    ap.add_argument("-i", "--input", type=int, required=False, default=parameter.input_width,
                    help="length of input seq.")
    ap.add_argument("-s", "--shift", type=int, required=False, default=parameter.shifted_width,
                    help="length of shift seq.")
    ap.add_argument("-o", "--output", type=int, required=False, default=parameter.label_width,
                    help="length of output seq.")
    ap.add_argument("-r", "--sample_rate", type=int, required=False, default=parameter.sample_rate,
                    help="sample rate when generating training sequence")
    ap.add_argument("-tr", "--test_sample_rate", type=int, required=False, default=parameter.test_sample_rate,
                    help="sample rate when generating testing sequence")
    ap.add_argument("-m", "--test_month", type=int, required=False, default=parameter.test_month,
                    help="month for testing")
    ap.add_argument("-d", "--dataset", type=str, required=False, default=parameter.csv_name,
                    help="name of dataset")
    ap.add_argument("-sm", "--split_mode", required=False, default=parameter.split_mode,
                    help="dataset splitting mode : {}".format(datautil.split_mode_list))
    ap.add_argument("-fn", "--feat_norm", required=False, default=parameter.norm_mode,
                    help="norm mode for feature column: {}".format(datautil.norm_type_list))
    ap.add_argument("-tn", "--target_norm", required=False, default=parameter.label_norm_mode,
                    help="norm mode for target column: {}".format(datautil.norm_type_list))
    ap.add_argument("-bs", "--batch_size", type=int, required=False, default=parameter.batchsize,
                    help="batch size")
    ap.add_argument("--shuffle", required=False, default=parameter.is_using_shuffle, action='store_true',
                    help='shuffle dataset or not')
    ap.add_argument("-by", "--bypass", type=int, required=False, default=parameter.bypass,
                    help="bypass mode: {}".format(bypass_factory.bypass_list))
    ap.add_argument("-te", "--time_embedding", type=int, required=False, default=parameter.time_embedding,
                    help="time embedding mode: {}".format(time_embedding_factory.time_embedding_list))
    ap.add_argument("-sd", '--split_day', required=False, default=parameter.split_days, action='store_true',
                    help='using split-days module or not')
    ap.add_argument('--use_image', required=False, default=parameter.is_using_image_data, action='store_true',
                    help='using image data as feature or not')
    ap.add_argument('--image_length', type=int, required=False, default=parameter.image_input_width3D,
                    help='length(on time axis) of images')
    ap.add_argument('--save_plot', required=False, default=parameter.save_plot, action='store_true',
                    help='save plot figure as html or not')
    ap.add_argument('--save_csv', required=False, default=parameter.save_csv, action='store_true',
                    help='save prediction as csv or not')

    args = vars(ap.parse_args())
    parameter.input_width = args["input"]
    parameter.shifted_width = args["shift"]
    parameter.label_width = args["output"]
    parameter.sample_rate = args["sample_rate"]
    parameter.test_sample_rate = args["test_sample_rate"]
    parameter.test_month = args["test_month"]
    parameter.csv_name = args["dataset"]
    parameter.features, parameter.target = parameter.set_dataset_related_params(args['dataset'])
    parameter.split_mode = args["split_mode"]
    parameter.norm_mode = args["feat_norm"]
    parameter.label_norm_mode = args["target_norm"]
    parameter.batchsize = args["batch_size"]
    parameter.is_using_shuffle = args["shuffle"]
    parameter.experiment_label = args["experiment_label"]
    parameter.bypass = args["bypass"]
    parameter.time_embedding = args["time_embedding"]
    parameter.split_days = args["split_day"]
    parameter.is_using_image_data = args["use_image"]
    parameter.image_input_width3D = args["image_length"]
    parameter.save_plot = args["save_plot"]
    parameter.save_csv = args["save_csv"]

    parameter.experiment_label += "_bypass-{}_TE-{}_split-{}".format(
        bypass_factory.BypassFac.get_bypass_mode(parameter.bypass),
        time_embedding_factory.TEFac.get_te_mode(parameter.time_embedding),
        parameter.split_days)
    # Initialise logging
    log = Msglog.LogInit(parameter.experiment_label, "logs/{}".format(parameter.experiment_label), 10, True, True)

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
                                      label_norm_mode=parameter.label_norm_mode,
                                      label_col=parameter.target,
                                      feature_col=parameter.features,
                                      split_mode=parameter.split_mode,
                                      month_sep=parameter.test_month,
                                      using_images=parameter.is_using_image_data)

    data_for_baseline = DataUtil(train_path=train_path,
                                 val_path=val_path,
                                 test_path=test_path,
                                 normalise=parameter.norm_mode,
                                 label_norm_mode=parameter.label_norm_mode,
                                 label_col=parameter.target,
                                 feature_col=parameter.target,
                                 split_mode=parameter.split_mode,
                                 month_sep=parameter.test_month)

    # windows generator#########################################################################################################################
    modelMetricsRecorder = {}
    if parameter.timeseries:
        dataUtil = data_with_weather_info
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
        log.info("\n------IO Setup------")
        log.info("input width: {}".format(input_width))
        log.info("shift width: {}".format(shift))
        log.info("label width: {}".format(label_width))
        if parameter.is_using_image_data:
            log.info("images width: {}".format(image_input_width))
        if "MA" in parameter.model_list:
            log.info("MA width: {}".format(MA_width))

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
                             samples_per_day=dataUtil.samples_per_day)
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

    #############################################################
    log = logging.getLogger(parameter.experiment_label)
    w = w2
    is_input_continuous_with_output = (shift == 0) and (not parameter.between8_17 or w.is_sampling_within_day)
    metrics_path = "plot/{}/{}".format(parameter.experiment_label, "all_metric")

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
        metrics_path = "plot/{}/{}".format(parameter.experiment_label, "all_metric")
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
        metrics_path = "plot/{}/{}".format(parameter.experiment_label, "all_metric")
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    # test learning models
    dataUtil = data_with_weather_info
    w = w2

    # pure numerical models
    if "convGRU" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("convGRU")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            input_scalar = Input(shape=(input_width, len(parameter.features)))

            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.time_embedding,
                                                                       tar_dim=model_convGRU.Config.embedding_filters,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.split_days or (not w.is_sampling_within_day and parameter.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=w.samples_per_day,
                                              in_dim=len(parameter.features),
                                              out_seq_len=label_width, out_dim=len(parameter.target),
                                              units=model_convGRU.Config.gru_units,
                                              filters=model_convGRU.Config.embedding_filters,
                                              kernel_size=model_convGRU.Config.embedding_kernel_size,
                                              gen_mode='unistep',
                                              is_seq_continuous=is_input_continuous_with_output,
                                              rate=model_convGRU.Config.dropout_rate)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(filters=model_convGRU.Config.embedding_filters,
                                                                filter_size=preprocess_utils.Config.kernel_size,
                                                                n_days=n_days,
                                                                n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_convGRU.ConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=input_width,
                                              in_dim=len(parameter.features),
                                              out_seq_len=label_width, out_dim=len(parameter.target),
                                              units=model_convGRU.Config.gru_units,
                                              filters=model_convGRU.Config.embedding_filters,
                                              kernel_size=model_convGRU.Config.embedding_kernel_size,
                                              gen_mode='unistep',
                                              is_seq_continuous=is_input_continuous_with_output,
                                              rate=model_convGRU.Config.dropout_rate)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.bypass,
                                                                out_width=label_width,
                                                                order=model_AR.Config.order,
                                                                in_dim=len(parameter.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)
            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name="convGRU")
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name="convGRU")

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
            input_scalar = Input(shape=(input_width, len(parameter.features)))
            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.time_embedding,
                                                                       tar_dim=model_convGRU.Config.embedding_filters,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.split_days or (not w.is_sampling_within_day and parameter.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                      d_model=model_transformer.Config.d_model,
                                                      num_heads=model_transformer.Config.n_heads,
                                                      dff=model_transformer.Config.dff,
                                                      src_seq_len=w.samples_per_day,
                                                      tar_seq_len=label_width,
                                                      src_dim=preprocess_utils.Config.filters,
                                                      tar_dim=len(parameter.target),
                                                      kernel_size=model_transformer.Config.embedding_kernel_size,
                                                      rate=model_transformer.Config.dropout_rate,
                                                      gen_mode="unistep",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False, token_len=0)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(filters=model_convGRU.Config.embedding_filters,
                                                                filter_size=preprocess_utils.Config.kernel_size,
                                                                n_days=n_days,
                                                                n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_transformer.Transformer(num_layers=model_transformer.Config.layers,
                                                      d_model=model_transformer.Config.d_model,
                                                      num_heads=model_transformer.Config.n_heads,
                                                      dff=model_transformer.Config.dff,
                                                      src_seq_len=input_width,
                                                      tar_seq_len=label_width, src_dim=len(parameter.features),
                                                      tar_dim=len(parameter.target),
                                                      kernel_size=model_transformer.Config.embedding_kernel_size,
                                                      rate=model_transformer.Config.dropout_rate,
                                                      gen_mode="unistep",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False, token_len=token_len)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)
            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.bypass,
                                                                out_width=label_width,
                                                                order=model_AR.Config.order,
                                                                in_dim=len(parameter.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)

            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name="transformer")
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name="transformer")
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

    if "stationary_convGRU" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("stationary_convGRU")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            input_scalar = Input(shape=(input_width, len(parameter.features)))
            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.time_embedding,
                                                                       tar_dim=model_convGRU.Config.embedding_filters,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.split_days or (not w.is_sampling_within_day and parameter.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_convGRU.StationaryConvGRU(num_layers=model_convGRU.Config.layers,
                                                        in_seq_len=w.samples_per_day,
                                                        in_dim=len(parameter.features),
                                                        out_seq_len=label_width, out_dim=len(parameter.target),
                                                        units=model_convGRU.Config.gru_units,
                                                        filters=model_convGRU.Config.embedding_filters,
                                                        kernel_size=model_convGRU.Config.embedding_kernel_size,
                                                        gen_mode='unistep',
                                                        is_seq_continuous=is_input_continuous_with_output,
                                                        rate=model_convGRU.Config.dropout_rate,
                                                        avg_window=series_decomposition.Config.window_size)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(filters=model_convGRU.Config.embedding_filters,
                                                                filter_size=preprocess_utils.Config.kernel_size,
                                                                n_days=n_days,
                                                                n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_convGRU.StationaryConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=input_width,
                                                        in_dim=len(parameter.features),
                                                        out_seq_len=label_width, out_dim=len(parameter.target),
                                                        units=model_convGRU.Config.gru_units,
                                                        filters=model_convGRU.Config.embedding_filters,
                                                        kernel_size=model_convGRU.Config.embedding_kernel_size,
                                                        gen_mode='unistep',
                                                        is_seq_continuous=is_input_continuous_with_output,
                                                        rate=model_convGRU.Config.dropout_rate,
                                                        avg_window=series_decomposition.Config.window_size)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.bypass,
                                                                out_width=label_width,
                                                                order=model_AR.Config.order,
                                                                in_dim=len(parameter.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)
            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name="stationary_convGRU")
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name="stationary_convGRU")
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="stationary_convGRU")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by stationary_convGRU...")

        metricsDict = w.allPlot(model=[best_model],
                                name="stationary_convGRU",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["stationary_convGRU"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "stationary_transformer" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("training stationary_transformer model...")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            if w.is_sampling_within_day:
                token_len = input_width
            else:
                token_len = (min(input_width, label_width) // w.samples_per_day // 2 + 1) * w.samples_per_day

            input_scalar = Input(shape=(input_width, len(parameter.features)))
            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.time_embedding,
                                                                       tar_dim=model_convGRU.Config.embedding_filters,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.split_days or (not w.is_sampling_within_day and parameter.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_transformer.StationaryTransformer(num_layers=model_transformer.Config.layers,
                                                                d_model=model_transformer.Config.d_model,
                                                                num_heads=model_transformer.Config.n_heads,
                                                                dff=model_transformer.Config.dff,
                                                                src_seq_len=w.samples_per_day,
                                                                tar_seq_len=label_width,
                                                                src_dim=preprocess_utils.Config.filters,
                                                                tar_dim=len(parameter.target),
                                                                kernel_size=model_transformer.Config.embedding_kernel_size,
                                                                rate=model_transformer.Config.dropout_rate,
                                                                gen_mode="unistep",
                                                                is_seq_continuous=is_input_continuous_with_output,
                                                                is_pooling=False, token_len=0,
                                                                avg_window=series_decomposition.Config.window_size)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(filters=model_convGRU.Config.embedding_filters,
                                                                filter_size=preprocess_utils.Config.kernel_size,
                                                                n_days=n_days,
                                                                n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_transformer.StationaryTransformer(num_layers=model_transformer.Config.layers,
                                                                d_model=model_transformer.Config.d_model,
                                                                num_heads=model_transformer.Config.n_heads,
                                                                dff=model_transformer.Config.dff,
                                                                src_seq_len=input_width,
                                                                tar_seq_len=label_width,
                                                                src_dim=len(parameter.features),
                                                                tar_dim=len(parameter.target),
                                                                kernel_size=model_transformer.Config.embedding_kernel_size,
                                                                rate=model_transformer.Config.dropout_rate,
                                                                gen_mode="unistep",
                                                                is_seq_continuous=is_input_continuous_with_output,
                                                                is_pooling=False, token_len=token_len,
                                                                avg_window=series_decomposition.Config.window_size)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.bypass,
                                                                out_width=label_width,
                                                                order=model_AR.Config.order,
                                                                in_dim=len(parameter.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)
            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs,
                                       name="stationary_transformer")
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name="stationary_transformer")
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="stationary_transformer")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by stationary_transformer...")

        metricsDict = w.allPlot(model=[best_model],
                                name="stationary_transformer",
                                scaler=dataUtil.labelScaler,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["stationary_transformer"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "znorm_convGRU" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("znorm_convGRU")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            input_scalar = Input(shape=(input_width, len(parameter.features)))

            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.time_embedding,
                                                                       tar_dim=model_convGRU.Config.embedding_filters,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.split_days or (not w.is_sampling_within_day and parameter.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_convGRU.MovingZNormConvGRU(num_layers=model_convGRU.Config.layers,
                                                         in_seq_len=w.samples_per_day,
                                                         in_dim=len(parameter.features),
                                                         out_seq_len=label_width, out_dim=len(parameter.target),
                                                         units=model_convGRU.Config.gru_units,
                                                         filters=model_convGRU.Config.embedding_filters,
                                                         kernel_size=model_convGRU.Config.embedding_kernel_size,
                                                         gen_mode='unistep',
                                                         is_seq_continuous=is_input_continuous_with_output,
                                                         rate=model_convGRU.Config.dropout_rate,
                                                         avg_window=series_decomposition.Config.window_size)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(filters=model_convGRU.Config.embedding_filters,
                                                                filter_size=preprocess_utils.Config.kernel_size,
                                                                n_days=n_days,
                                                                n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_convGRU.MovingZNormConvGRU(num_layers=model_convGRU.Config.layers, in_seq_len=input_width,
                                                         in_dim=len(parameter.features),
                                                         out_seq_len=label_width, out_dim=len(parameter.target),
                                                         units=model_convGRU.Config.gru_units,
                                                         filters=model_convGRU.Config.embedding_filters,
                                                         kernel_size=model_convGRU.Config.embedding_kernel_size,
                                                         gen_mode='unistep',
                                                         is_seq_continuous=is_input_continuous_with_output,
                                                         rate=model_convGRU.Config.dropout_rate,
                                                         avg_window=series_decomposition.Config.window_size)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.bypass,
                                                                out_width=label_width,
                                                                order=model_AR.Config.order,
                                                                in_dim=len(parameter.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)
            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name="znorm_convGRU")
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name="znorm_convGRU")
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="znorm_convGRU")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by znorm_convGRU...")

        metricsDict = w.allPlot(model=[best_model],
                                name="znorm_convGRU",
                                scaler=dataUtil.labelScaler,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["znorm_convGRU"] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "znorm_transformer" in parameter.model_list:
        best_perform, best_perform2 = None, None
        best_model, best_model2 = None, None
        log.info("training znorm_transformer model...")
        for testEpoch in parameter.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            if w.is_sampling_within_day:
                token_len = input_width
            else:
                token_len = (min(input_width, label_width) // w.samples_per_day // 2 + 1) * w.samples_per_day

            input_scalar = Input(shape=(input_width, len(parameter.features)))
            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.time_embedding,
                                                                       tar_dim=model_convGRU.Config.embedding_filters,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.split_days or (not w.is_sampling_within_day and parameter.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=preprocess_utils.Config.filters,
                                                        filter_size=preprocess_utils.Config.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_transformer.MovingZScoreNormTransformer(num_layers=model_transformer.Config.layers,
                                                                      d_model=model_transformer.Config.d_model,
                                                                      num_heads=model_transformer.Config.n_heads,
                                                                      dff=model_transformer.Config.dff,
                                                                      src_seq_len=w.samples_per_day,
                                                                      tar_seq_len=label_width,
                                                                      src_dim=preprocess_utils.Config.filters,
                                                                      tar_dim=len(parameter.target),
                                                                      kernel_size=model_transformer.Config.embedding_kernel_size,
                                                                      rate=model_transformer.Config.dropout_rate,
                                                                      gen_mode="unistep",
                                                                      is_seq_continuous=is_input_continuous_with_output,
                                                                      is_pooling=False, token_len=0,
                                                                      avg_window=series_decomposition.Config.window_size)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(filters=model_convGRU.Config.embedding_filters,
                                                                filter_size=preprocess_utils.Config.kernel_size,
                                                                n_days=n_days,
                                                                n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_transformer.MovingZScoreNormTransformer(num_layers=model_transformer.Config.layers,
                                                                      d_model=model_transformer.Config.d_model,
                                                                      num_heads=model_transformer.Config.n_heads,
                                                                      dff=model_transformer.Config.dff,
                                                                      src_seq_len=input_width,
                                                                      tar_seq_len=label_width,
                                                                      src_dim=len(parameter.features),
                                                                      tar_dim=len(parameter.target),
                                                                      kernel_size=model_transformer.Config.embedding_kernel_size,
                                                                      rate=model_transformer.Config.dropout_rate,
                                                                      gen_mode="unistep",
                                                                      is_seq_continuous=is_input_continuous_with_output,
                                                                      is_pooling=False, token_len=token_len,
                                                                      avg_window=series_decomposition.Config.window_size)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.bypass,
                                                                out_width=label_width,
                                                                order=model_AR.Config.order,
                                                                in_dim=len(parameter.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)
            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name="znorm_transformer")
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name="znorm_transformer")
            datamodel_CL, datamodel_CL_performance = ModelTrainer(dataGnerator=w, model=model,
                                                                  generatorMode="data", testEpoch=testEpoch,
                                                                  name="znorm_transformer")
            print(datamodel_CL_performance)
            if ((best_perform == None) or (best_perform[3] > datamodel_CL_performance[3])):
                best_model = datamodel_CL
                best_perform = datamodel_CL_performance
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by znorm_transformer...")

        metricsDict = w.allPlot(model=[best_model],
                                name="znorm_transformer",
                                scaler=dataUtil.labelScaler,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["znorm_transformer"] = metricsDict[logM]
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

    metrics_path = "plot/{}/{}".format(parameter.experiment_label, "all_metric")
    pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    return modelMetricsRecorder


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(run_eagerly=True)
    # tf.data.experimental.enable_debug_mode()
    result = run()
