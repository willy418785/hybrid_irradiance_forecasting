# python main.py -d  ../skyImage
# import the necessary packages
import datetime
import json

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

def ModelTrainer(dataGnerator: WindowGenerator, model, sample_rate, generatorMode="", testEpoch=0, name="Example",
                 is_shuffle=False):
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                  , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()
            , my_metrics.VWMAPE
            , my_metrics.root_relative_squared_error
            , my_metrics.corr])
    model.summary()
    tf.keras.backend.clear_session()
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    using_timestamp_data = time_embedding_factory.TEFac.get_te_mode(
        parameter.model_params.time_embedding) is not None and name not in parameter.exp_params.baselines
    if generatorMode == "combined" or generatorMode == "data":
        history = model.fit(
            dataGnerator.train(sample_rate, addcloud=parameter.data_params.addAverage,
                               using_timestamp_data=using_timestamp_data,
                               is_shuffle=is_shuffle),
            validation_data=dataGnerator.val(sample_rate,
                                             addcloud=parameter.data_params.addAverage,
                                             using_timestamp_data=using_timestamp_data,
                                             is_shuffle=is_shuffle),
            epochs=testEpoch, batch_size=parameter.exp_params.batch_size, callbacks=parameter.exp_params.callbacks)

    elif generatorMode == "image":
        history = model.fit(dataGnerator.trainWithArg, validation_data=dataGnerator.valWithArg,
                            epochs=testEpoch, batch_size=parameter.exp_params.batch_size,
                            callbacks=parameter.exp_params.callbacks)

    return model, history


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
                   epochs=testEpoch, batch_size=parameter.exp_params.batch_size,
                   callbacks=parameter.exp_params.callbacks)
        A_pred, A_y = dataGnerator.plotPredictUnit(model1, dataGnerator.valAC(sepMode="cloudA"), datamode=generatorMode)

        model2.fit(dataGnerator.trainAC(sepMode="cloudC"), validation_data=dataGnerator.valAC(sepMode="cloudC"),
                   epochs=testEpoch, batch_size=parameter.exp_params.batch_size,
                   callbacks=parameter.exp_params.callbacks)
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
                   epochs=testEpoch, batch_size=parameter.exp_params.batch_size,
                   callbacks=parameter.exp_params.callbacks)
        A_pred, A_y = dataGnerator.plotPredictUnit(model1, dataGnerator.valDataAC(sepMode="cloudA"),
                                                   datamode=generatorMode)
        # print(A_pred, A_y)

        model2.fit(dataGnerator.trainDataAC(sepMode="cloudC"), validation_data=dataGnerator.valDataAC(sepMode="cloudC"),
                   epochs=testEpoch, batch_size=parameter.exp_params.batch_size,
                   callbacks=parameter.exp_params.callbacks)
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


def args_parse():
    def cast_int_if_possible(arg):
        if arg.isdecimal():
            return int(arg)
        else:
            return arg

    ap = argparse.ArgumentParser()

    # experiment related arguments
    ap.add_argument("-n", "--experiment_label", type=str, required=True, default=parameter.exp_params.experiment_label,
                    help="experiment_label")
    ap.add_argument("-bs", "--batch_size", type=int, required=False, default=parameter.exp_params.batch_size,
                    help="batch size")
    ap.add_argument('--save_plot', required=False, default=parameter.exp_params.save_plot, action='store_true',
                    help='save plot figure as html or not')
    ap.add_argument('--save_csv', required=False, default=parameter.exp_params.save_csv, action='store_true',
                    help='save prediction as csv or not')
    ap.add_argument('--model_selection', nargs='*', type=cast_int_if_possible, required=False, default='default',
                    help="models selection mode: {}".format(parameter.exp_params.model_selection_mode))
    ap.add_argument('--epoch_list', nargs='*', type=int, required=False, default=parameter.exp_params.epoch_list,
                    help="list of number of epoch for each independent training process")

    # data related arguments
    ap.add_argument("-i", "--input", type=int, required=False, default=parameter.data_params.input_width,
                    help="length of input seq.")
    ap.add_argument("-s", "--shift", type=int, required=False, default=parameter.data_params.shifted_width,
                    help="length of shift seq.")
    ap.add_argument("-o", "--output", type=int, required=False, default=parameter.data_params.label_width,
                    help="length of output seq.")
    ap.add_argument("-r", "--sample_rate", type=int, required=False, default=parameter.data_params.sample_rate,
                    help="sample rate when generating training sequence")
    ap.add_argument("-tr", "--test_sample_rate", type=int, required=False,
                    default=parameter.data_params.test_sample_rate,
                    help="sample rate when generating testing sequence")
    ap.add_argument("-m", "--test_month", type=int, required=False, default=parameter.data_params.test_month,
                    help="month for testing")
    ap.add_argument("-d", "--dataset", type=str, required=False, default=parameter.data_params.csv_name,
                    help="name of dataset")
    ap.add_argument("-sm", "--split_mode", type=cast_int_if_possible, required=False,
                    default=parameter.data_params.split_mode,
                    help="dataset splitting mode : {}".format(datautil.split_mode_list))
    ap.add_argument("-fn", "--feat_norm", type=cast_int_if_possible, required=False,
                    default=parameter.data_params.norm_mode,
                    help="norm mode for feature column: {}".format(datautil.norm_type_list))
    ap.add_argument("-tn", "--target_norm", type=cast_int_if_possible, required=False,
                    default=parameter.data_params.label_norm_mode,
                    help="norm mode for target column: {}".format(datautil.norm_type_list))
    ap.add_argument("--shuffle", required=False, default=parameter.data_params.is_using_shuffle, action='store_true',
                    help='shuffle dataset or not')
    ap.add_argument('--use_image', required=False, default=parameter.data_params.is_using_image_data,
                    action='store_true',
                    help='using image data as feature or not')
    ap.add_argument('--image_length', type=int, required=False, default=parameter.data_params.image_input_width3D,
                    help='length(on time axis) of images')

    # model related arguments
    ap.add_argument("-by", "--bypass", type=cast_int_if_possible, required=False, default=parameter.model_params.bypass,
                    help="bypass mode: {}".format(bypass_factory.bypass_list))
    ap.add_argument("-te", "--time_embedding", type=cast_int_if_possible, required=False,
                    default=parameter.model_params.time_embedding,
                    help="time embedding mode: {}".format(time_embedding_factory.time_embedding_list))
    ap.add_argument("-sd", '--split_day', required=False, default=parameter.model_params.split_days,
                    action='store_true',
                    help='using split-days module or not')
    # general model arguments
    ap.add_argument("--layers", type=int, required=False, default=None, help="number of layers")
    ap.add_argument("--kernel_size", type=int, required=False, default=None, help="kernel's size of Conv1D")
    ap.add_argument("--dropout", type=float, required=False, default=None, help="dropout rate")
    # transformer model arguments
    ap.add_argument("--d_model", type=int, required=False, default=parameter.model_params.transformer_params.d_model,
                    help="number of inner dimensions of transformer model")
    ap.add_argument("--n_heads", type=int, required=False, default=parameter.model_params.transformer_params.n_heads,
                    help="number of heads of transformer model")
    ap.add_argument("--dff", type=int, required=False, default=parameter.model_params.transformer_params.dff,
                    help="number of intermediate dimensions of feed-forward layer of transformer model")
    ap.add_argument("--token", type=int, required=False, default=parameter.model_params.transformer_params.token_length,
                    help="length of token which is used as a part of decoder input")
    # convGRU model arguments
    ap.add_argument("--filters", type=int, required=False,
                    default=parameter.model_params.convGRU_params.embedding_filters,
                    help="number of filters of Conv1D for convGRU model")
    ap.add_argument("--units", type=int, required=False, default=parameter.model_params.convGRU_params.gru_units,
                    help="number of units of convGRU model")
    # bypass LR arguments
    ap.add_argument("--order", type=int, required=False, default=parameter.model_params.bypass_params.order,
                    help="order of bypass LR")
    # series decomposition arguments
    ap.add_argument("--window", type=int, required=False, default=parameter.model_params.bypass_params.order,
                    help="size of moving average window used in series decomposition block")

    args = vars(ap.parse_args())
    # exp. related params assignment
    parameter.exp_params.experiment_label = args["experiment_label"]
    parameter.exp_params.batch_size = args["batch_size"]
    parameter.exp_params.save_plot = args["save_plot"]
    parameter.exp_params.save_csv = args["save_csv"]
    model_selection = parameter.exp_params.set_tested_models(args["model_selection"])
    parameter.exp_params.epoch_list = args["epoch_list"]
    # data related params assignment
    parameter.data_params.input_width = args["input"]
    parameter.data_params.shifted_width = args["shift"]
    parameter.data_params.label_width = args["output"]
    parameter.data_params.sample_rate = args["sample_rate"]
    parameter.data_params.test_sample_rate = args["test_sample_rate"]
    parameter.data_params.test_month = args["test_month"]
    parameter.data_params.csv_name = args["dataset"]
    parameter.data_params.split_mode = args["split_mode"]
    parameter.data_params.norm_mode = args["feat_norm"]
    parameter.data_params.label_norm_mode = args["target_norm"]
    parameter.data_params.is_using_shuffle = args["shuffle"]
    parameter.data_params.is_using_image_data = args["use_image"]
    parameter.data_params.image_input_width3D = args["image_length"]
    # dynamic data related params adjustment
    parameter.data_params.set_dataset_params()
    parameter.data_params.set_start_end_time()
    parameter.data_params.set_image_params()
    # model related params assignment
    parameter.model_params.bypass = args["bypass"]
    parameter.model_params.time_embedding = args["time_embedding"]
    parameter.model_params.split_days = args["split_day"]
    # set optimized hyper parameters corresponding to different dataset
    parameter.model_params.set_ideal(args["dataset"])
    # universal params of each models
    if args["layers"] is not None:
        parameter.model_params.transformer_params.layers = args["layers"]
        parameter.model_params.convGRU_params.layers = args["layers"]
    if args["kernel_size"] is not None:
        parameter.model_params.transformer_params.embedding_kernel_size = args["kernel_size"]
        parameter.model_params.convGRU_params.embedding_kernel_size = args["kernel_size"]
    if args["dropout"] is not None:
        parameter.model_params.transformer_params.dropout_rate = args["dropout"]
        parameter.model_params.convGRU_params.dropout_rate = args["dropout"]
    # transformer params
    parameter.model_params.transformer_params.d_model = args["d_model"]
    parameter.model_params.transformer_params.n_heads = args["n_heads"]
    parameter.model_params.transformer_params.dff = args["dff"]
    parameter.model_params.transformer_params.token_length = args["token"]
    # convGRU params
    parameter.model_params.convGRU_params.embedding_filters = args["filters"]
    parameter.model_params.convGRU_params.gru_units = args["units"]
    # bypass module params
    parameter.model_params.bypass_params.order = args["order"]
    # decomposition module params
    parameter.model_params.decompose_params.avg_window = args["window"]
    # dynamic model params adjustment
    parameter.model_params.transformer_params.adjust(
        parameter.data_params.input_width)  # adjust token length of transformer
    parameter.model_params.bypass_params.adjust(parameter.data_params.input_width)  # adjust order of bypass LR

    # format directory name of this experiment
    parameter.exp_params.experiment_label += "_{}".format(model_selection)
    file_name, _ = os.path.splitext(parameter.data_params.csv_name)
    parameter.exp_params.experiment_label += "_{}".format(file_name)
    parameter.exp_params.experiment_label += "_i{}s{}o{}".format(
        parameter.data_params.input_width,
        parameter.data_params.shifted_width,
        parameter.data_params.label_width)
    parameter.exp_params.experiment_label += "_rate{}trate{}".format(
        parameter.data_params.sample_rate,
        parameter.data_params.test_sample_rate)
    parameter.exp_params.experiment_label += "_norm[{}]scale[{}]".format(
        datautil.get_mode(parameter.data_params.norm_mode, datautil.norm_type_list),
        datautil.get_mode(parameter.data_params.label_norm_mode, datautil.norm_type_list))
    if not set(parameter.exp_params.model_list).issubset(set(parameter.exp_params.baselines)):
        # add setup info to dir name if not testing baseline methods only
        parameter.exp_params.experiment_label += "_bypass[{}]TE[{}]split[{}]".format(
            bypass_factory.BypassFac.get_bypass_mode(parameter.model_params.bypass),
            time_embedding_factory.TEFac.get_te_mode(parameter.model_params.time_embedding),
            parameter.model_params.split_days)
    if get_mode(parameter.data_params.split_mode, datautil.split_mode_list) != "all_year":
        parameter.exp_params.experiment_label += "_datasplit[{}]_test_on_{}".format(
            get_mode(parameter.data_params.split_mode, datautil.split_mode_list),
            datetime.datetime.strptime(str(parameter.data_params.test_month), "%m").strftime("%b"))
    return args


def run():
    # Initialize logging
    log = Msglog.LogInit(parameter.exp_params.experiment_label, "logs/{}".format(parameter.exp_params.experiment_label),
                         10, True, True)
    log.info("\n######Current Configuration######\n{}{}{}".format(parameter.data_params,
                                                                  parameter.exp_params,
                                                                  parameter.model_params))

    log.info("Python version: %s", sys.version)
    log.info("Tensorflow version: %s", tf.__version__)
    log.info("Keras version: %s ... Using tensorflow embedded keras", tf.keras.__version__)

    # log.info("static_suffle: {}".format(parameter.static_suffle))
    # log.info("dynamic_suffle: {}".format(parameter.dynamic_suffle))
    log.info("input features: {}".format(parameter.data_params.features))
    log.info("targets: {}".format(parameter.data_params.target))
    if parameter.data_params.is_using_image_data:
        log.info("image_input_width: {}".format(parameter.data_params.image_input_width3D))
        log.info("image_depth: {}".format(parameter.data_params.image_depth))
    log.info("batchsize: {}".format(parameter.exp_params.batch_size))
    log.info("callbacks: {}".format(parameter.exp_params.callbacks))
    log.info("model_list: {}".format(parameter.exp_params.model_list))
    # log.info("class_type: {}".format(parameter.class_type))
    log.info("epoch list: {}".format(parameter.exp_params.epoch_list))
    log.info("split mode: {}".format(parameter.data_params.split_mode))
    log.info("test month: {}".format(parameter.data_params.test_month))
    log.info("data input: {}".format(parameter.inputs))
    log.info("Add sun_average: {}".format(parameter.data_params.addAverage))
    log.info("Model nums: {}".format(parameter.dynamic_model))
    log.info("Using shuffle: {}".format(parameter.data_params.is_using_shuffle))
    log.info("smoothing type: {}".format(parameter.data_params.smoothing_type))
    log.info("csv file name: {}".format(parameter.data_params.csv_name))
    log.info("only using daytime data: {}".format(parameter.data_params.between8_17))
    log.info("only evaluate daytime prediction: {}".format(parameter.data_params.test_between8_17))
    log.info("time granularity: {}".format(parameter.data_params.time_granularity))

    # construct the path to the input .txt file that contains information
    # on each house in the dataset and then load the dataset
    log.info("loading cloud attributes...")
    train_path = os.path.sep.join([parameter.data_params.csv_name])
    val_path = None
    test_path = None

    data_with_weather_info = DataUtil(train_path=train_path,
                                      val_path=val_path,
                                      test_path=test_path,
                                      normalise=parameter.data_params.norm_mode,
                                      label_norm_mode=parameter.data_params.label_norm_mode,
                                      label_col=parameter.data_params.target,
                                      feature_col=parameter.data_params.features,
                                      split_mode=parameter.data_params.split_mode,
                                      month_sep=parameter.data_params.test_month,
                                      using_images=parameter.data_params.is_using_image_data,
                                      smoothing_mode=parameter.data_params.smoothing_type,
                                      smoothing_parameter=parameter.data_params.smoothing_parameter)

    data_for_baseline = DataUtil(train_path=train_path,
                                 val_path=val_path,
                                 test_path=test_path,
                                 normalise=parameter.data_params.norm_mode,
                                 label_norm_mode=parameter.data_params.label_norm_mode,
                                 label_col=parameter.data_params.target,
                                 feature_col=parameter.data_params.target,
                                 split_mode=parameter.data_params.split_mode,
                                 month_sep=parameter.data_params.test_month,
                                 smoothing_mode=parameter.data_params.smoothing_type,
                                 smoothing_parameter=parameter.data_params.smoothing_parameter)

    # windows generator#########################################################################################################################
    modelMetricsRecorder = {}
    dataUtil = data_with_weather_info
    assert type(parameter.data_params.input_width) is int
    assert type(parameter.data_params.shifted_width) is int
    assert type(parameter.data_params.label_width) is int
    if parameter.data_params.is_using_image_data:
        assert type(parameter.data_params.image_input_width3D) is int
        image_input_width = parameter.data_params.image_input_width3D
    else:
        image_input_width = 0
    if "MA" in parameter.exp_params.model_list:
        assert type(parameter.data_params.MA_width) is int
        MA_width = parameter.data_params.MA_width
    input_width = parameter.data_params.input_width
    shift = parameter.data_params.shifted_width
    label_width = parameter.data_params.label_width
    log.info("\n------IO Setup------")
    log.info("input width: {}".format(input_width))
    log.info("shift width: {}".format(shift))
    log.info("label width: {}".format(label_width))
    if parameter.data_params.is_using_image_data:
        log.info("images width: {}".format(image_input_width))
    if "MA" in parameter.exp_params.model_list:
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
                         batch_size=parameter.exp_params.batch_size,
                         label_columns="ShortWaveDown",
                         samples_per_day=dataUtil.samples_per_day)
    #############################################################
    log = logging.getLogger(parameter.exp_params.experiment_label)
    w = w2
    is_input_continuous_with_output = (shift == 0) and (
            not parameter.data_params.between8_17 or w.is_sampling_within_day)
    metrics_path = "plot/{}/{}".format(parameter.exp_params.experiment_label, "all_metric")

    # test baseline model
    dataUtil = data_for_baseline
    if "Persistence" in parameter.exp_params.model_list:
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
                                            batch_size=parameter.exp_params.batch_size,
                                            label_columns="ShortWaveDown",
                                            samples_per_day=dataUtil.samples_per_day)
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
                                                save_csv=parameter.exp_params.save_csv,
                                                save_plot=parameter.exp_params.save_plot,
                                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["Persistence"] = metricsDict[logM]
        metrics_path = "plot/{}/{}".format(parameter.exp_params.experiment_label, "all_metric")
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "MA" in parameter.exp_params.model_list:
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
                                   batch_size=parameter.exp_params.batch_size,
                                   label_columns="ShortWaveDown",
                                   samples_per_day=dataUtil.samples_per_day)
        movingAverage = MA(MA_width, w_for_MA.is_sampling_within_day, w_for_MA.samples_per_day, label_width)
        movingAverage.compile(loss=tf.losses.MeanSquaredError(),
                              metrics=[tf.metrics.MeanAbsoluteError(),
                                       tf.metrics.MeanAbsolutePercentageError(),
                                       my_metrics.VWMAPE,
                                       my_metrics.corr])
        metricsDict = w_for_MA.allPlot(model=[movingAverage],
                                       name="MA",
                                       scaler=dataUtil.labelScaler,
                                       save_csv=parameter.exp_params.save_csv,
                                       save_plot=parameter.exp_params.save_plot,
                                       datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM]["MA"] = metricsDict[logM]
        metrics_path = "plot/{}/{}".format(parameter.exp_params.experiment_label, "all_metric")
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "LR" in parameter.exp_params.model_list:
        w = w2
        model_name = "LR"
        log.info("training {} model...".format(model_name))
        testEpoch = parameter.exp_params.epoch_list[0]
        input_scalar = Input(shape=(input_width, len(parameter.data_params.features)))
        linear_regression = model_AR.TemporalChannelIndependentLR(order=parameter.model_params.bypass_params.order,
                                                                  tar_seq_len=label_width,
                                                                  src_dims=len(parameter.data_params.features))
        linear_regression = linear_regression(input_scalar)
        model = tf.keras.Model(inputs=input_scalar, outputs=linear_regression, name=model_name)
        datamodel_CL, history = ModelTrainer(dataGnerator=w, model=model, sample_rate=parameter.data_params.sample_rate,
                                             generatorMode="data", testEpoch=testEpoch, name=model_name)
        if 'val_loss' in history.history:
            with open('./plot/{}/history-{}.json'.format(parameter.exp_params.experiment_label, model_name),
                      'w') as f:
                json.dump(history.history, f, indent='\t')
        log.info("predicting SolarIrradiation by {}...".format(model_name))
        metricsDict = w.allPlot(model=[datamodel_CL],
                                name=model_name,
                                scaler=dataUtil.labelScaler,
                                save_csv=parameter.exp_params.save_csv,
                                save_plot=parameter.exp_params.save_plot,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM][model_name] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "AR" in parameter.exp_params.model_list:
        teacher_forcing_w = WindowGenerator(input_width=input_width,
                                            image_input_width=image_input_width,
                                            label_width=1,
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
                                            batch_size=parameter.exp_params.batch_size,
                                            label_columns="ShortWaveDown",
                                            samples_per_day=dataUtil.samples_per_day)
        model_name = "AR"
        log.info("training {} model...".format(model_name))
        testEpoch = parameter.exp_params.epoch_list[0]
        input_scalar = Input(shape=(input_width, len(parameter.data_params.features)))
        auto_regression = model_AR.ChannelIndependentAR(order=parameter.model_params.bypass_params.order,
                                                        src_dims=len(parameter.data_params.features))
        output = auto_regression(input_scalar)
        model = model_AR.ARModel(ar=auto_regression, tar_len=label_width, inputs=input_scalar, outputs=output)
        datamodel_CL, history = ModelTrainer(dataGnerator=teacher_forcing_w, model=model, sample_rate=1,
                                             generatorMode="data", testEpoch=testEpoch, name=model_name)
        if 'val_loss' in history.history:
            with open('./plot/{}/history-{}.json'.format(parameter.exp_params.experiment_label, model_name),
                      'w') as f:
                json.dump(history.history, f, indent='\t')
        log.info("predicting SolarIrradiation by {}...".format(model_name))
        evaluation_w = w2
        metricsDict = evaluation_w.allPlot(model=[datamodel_CL],
                                           name=model_name,
                                           scaler=dataUtil.labelScaler,
                                           save_csv=parameter.exp_params.save_csv,
                                           save_plot=parameter.exp_params.save_plot,
                                           datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM][model_name] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "LSTNet" in parameter.exp_params.model_list:
        teacher_forcing_w = WindowGenerator(input_width=input_width,
                                            image_input_width=image_input_width,
                                            label_width=1,
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
                                            batch_size=parameter.exp_params.batch_size,
                                            label_columns="ShortWaveDown",
                                            samples_per_day=dataUtil.samples_per_day)
        model_name = "LSTNet"
        best_perform, best_perform2 = float('inf'), float('inf')
        best_model, best_model2 = None, None
        log.info(model_name)
        testEpoch = parameter.exp_params.epoch_list[0]

        args_dict = GetArgumentsDict()
        init = LSTNetInit(args_dict, True)
        init.window = input_width
        init.skip = w.samples_per_day if w.samples_per_day < input_width else input_width
        init.highway = w.samples_per_day if w.samples_per_day < input_width else input_width
        model = LSTNetModel(init, (None, input_width, len(parameter.data_params.features)))
        model = model_AR.ARModel(ar=model, tar_len=label_width, inputs=model.inputs, outputs=model.outputs)
        datamodel_CL, history = ModelTrainer(dataGnerator=teacher_forcing_w, model=model, sample_rate=1,
                                             generatorMode="data",
                                             testEpoch=testEpoch, name="LSTNet",
                                             is_shuffle=parameter.data_params.is_using_shuffle)
        if 'val_loss' in history.history:
            with open('./plot/{}/history-{}.json'.format(parameter.exp_params.experiment_label, model_name),
                      'w') as f:
                json.dump(history.history, f, indent='\t')
        print(best_perform)
        log.info("a model ok")

        log.info("predicting SolarIrradiation by {}...".format(model_name))
        best_model = datamodel_CL if best_model is None else best_model
        evaluation_w = w2
        metricsDict = evaluation_w.allPlot(model=[best_model],
                                           name=model_name,
                                           scaler=dataUtil.labelScaler,
                                           save_csv=parameter.exp_params.save_csv,
                                           save_plot=parameter.exp_params.save_plot,
                                           datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM][model_name] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))
    # test learning models
    dataUtil = data_with_weather_info
    w = w2

    # pure numerical models
    if "convGRU" in parameter.exp_params.model_list:
        model_name = "convGRU"
        best_perform, best_perform2 = float('inf'), float('inf')
        best_model, best_model2 = None, None
        log.info(model_name)
        for testEpoch in parameter.exp_params.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            input_scalar = Input(shape=(input_width, len(parameter.data_params.features)))

            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.model_params.time_embedding,
                                                                       tar_dim=parameter.model_params.convGRU_params.embedding_filters,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.model_params.split_days or (
                    not w.is_sampling_within_day and parameter.data_params.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=parameter.model_params.split_day_params.filters,
                                                        filter_size=parameter.model_params.split_day_params.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_convGRU.ConvGRU(num_layers=parameter.model_params.convGRU_params.layers,
                                              in_seq_len=w.samples_per_day,
                                              in_dim=len(parameter.data_params.features),
                                              out_seq_len=label_width, out_dim=len(parameter.data_params.target),
                                              units=parameter.model_params.convGRU_params.gru_units,
                                              filters=parameter.model_params.convGRU_params.embedding_filters,
                                              kernel_size=parameter.model_params.convGRU_params.embedding_kernel_size,
                                              gen_mode='unistep',
                                              is_seq_continuous=is_input_continuous_with_output,
                                              rate=parameter.model_params.convGRU_params.dropout_rate)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(
                        filters=parameter.model_params.convGRU_params.embedding_filters,
                        filter_size=parameter.model_params.split_day_params.kernel_size,
                        n_days=n_days,
                        n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_convGRU.ConvGRU(num_layers=parameter.model_params.convGRU_params.layers,
                                              in_seq_len=input_width,
                                              in_dim=len(parameter.data_params.features),
                                              out_seq_len=label_width, out_dim=len(parameter.data_params.target),
                                              units=parameter.model_params.convGRU_params.gru_units,
                                              filters=parameter.model_params.convGRU_params.embedding_filters,
                                              kernel_size=parameter.model_params.convGRU_params.embedding_kernel_size,
                                              gen_mode='unistep',
                                              is_seq_continuous=is_input_continuous_with_output,
                                              rate=parameter.model_params.convGRU_params.dropout_rate)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.model_params.bypass,
                                                                out_width=label_width,
                                                                order=parameter.model_params.bypass_params.order,
                                                                in_dim=len(parameter.data_params.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)
            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name=model_name)
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name=model_name)

            datamodel_CL, history = ModelTrainer(dataGnerator=w, model=model,
                                                 sample_rate=parameter.data_params.sample_rate, generatorMode="data",
                                                 testEpoch=testEpoch, name=model_name,
                                                 is_shuffle=parameter.data_params.is_using_shuffle)

            if 'val_loss' in history.history and best_perform > min(history.history["val_loss"]):
                best_model = datamodel_CL
                best_perform = min(history.history["val_loss"])
                with open('./plot/{}/history-{}.json'.format(parameter.exp_params.experiment_label, model_name),
                          'w') as f:
                    json.dump(history.history, f, indent='\t')
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by {}...".format(model_name))
        best_model = datamodel_CL if best_model is None else best_model
        metricsDict = w.allPlot(model=[best_model],
                                name=model_name,
                                scaler=dataUtil.labelScaler,
                                save_csv=parameter.exp_params.save_csv,
                                save_plot=parameter.exp_params.save_plot,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM][model_name] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "transformer" in parameter.exp_params.model_list:
        model_name = "transformer"
        best_perform, best_perform2 = float('inf'), float('inf')
        best_model, best_model2 = None, None
        log.info("training {} model...".format(model_name))
        for testEpoch in parameter.exp_params.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            token_len = parameter.model_params.transformer_params.token_length
            input_scalar = Input(shape=(input_width, len(parameter.data_params.features)))
            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.model_params.time_embedding,
                                                                       tar_dim=parameter.model_params.transformer_params.d_model,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.model_params.split_days or (
                    not w.is_sampling_within_day and parameter.data_params.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=parameter.model_params.split_day_params.filters,
                                                        filter_size=parameter.model_params.split_day_params.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_transformer.Transformer(num_layers=parameter.model_params.transformer_params.layers,
                                                      d_model=parameter.model_params.transformer_params.d_model,
                                                      num_heads=parameter.model_params.transformer_params.n_heads,
                                                      dff=parameter.model_params.transformer_params.dff,
                                                      src_seq_len=w.samples_per_day,
                                                      tar_seq_len=label_width,
                                                      src_dim=parameter.model_params.split_day_params.filters,
                                                      tar_dim=len(parameter.data_params.target),
                                                      kernel_size=parameter.model_params.transformer_params.embedding_kernel_size,
                                                      rate=parameter.model_params.transformer_params.dropout_rate,
                                                      gen_mode="unistep",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False, token_len=0)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(
                        filters=parameter.model_params.convGRU_params.embedding_filters,
                        filter_size=parameter.model_params.split_day_params.kernel_size,
                        n_days=n_days,
                        n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_transformer.Transformer(num_layers=parameter.model_params.transformer_params.layers,
                                                      d_model=parameter.model_params.transformer_params.d_model,
                                                      num_heads=parameter.model_params.transformer_params.n_heads,
                                                      dff=parameter.model_params.transformer_params.dff,
                                                      src_seq_len=input_width,
                                                      tar_seq_len=label_width,
                                                      src_dim=len(parameter.data_params.features),
                                                      tar_dim=len(parameter.data_params.target),
                                                      kernel_size=parameter.model_params.transformer_params.embedding_kernel_size,
                                                      rate=parameter.model_params.transformer_params.dropout_rate,
                                                      gen_mode="unistep",
                                                      is_seq_continuous=is_input_continuous_with_output,
                                                      is_pooling=False, token_len=token_len)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)
            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.model_params.bypass,
                                                                out_width=label_width,
                                                                order=parameter.model_params.bypass_params.order,
                                                                in_dim=len(parameter.data_params.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)

            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name=model_name)
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name=model_name)
            datamodel_CL, history = ModelTrainer(dataGnerator=w, model=model,
                                                 sample_rate=parameter.data_params.sample_rate, generatorMode="data",
                                                 testEpoch=testEpoch, name=model_name,
                                                 is_shuffle=parameter.data_params.is_using_shuffle)
            if 'val_loss' in history.history and best_perform > min(history.history["val_loss"]):
                best_model = datamodel_CL
                best_perform = min(history.history["val_loss"])
                with open('./plot/{}/history-{}.json'.format(parameter.exp_params.experiment_label, model_name),
                          'w') as f:
                    json.dump(history.history, f, indent='\t')
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by {}...".format(model_name))
        best_model = datamodel_CL if best_model is None else best_model
        metricsDict = w.allPlot(model=[best_model],
                                name=model_name,
                                scaler=dataUtil.labelScaler,
                                save_csv=parameter.exp_params.save_csv,
                                save_plot=parameter.exp_params.save_plot,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM][model_name] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "stationary_convGRU" in parameter.exp_params.model_list:
        model_name = "stationary_convGRU"
        best_perform, best_perform2 = float('inf'), float('inf')
        best_model, best_model2 = None, None
        log.info(model_name)
        for testEpoch in parameter.exp_params.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            input_scalar = Input(shape=(input_width, len(parameter.data_params.features)))
            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.model_params.time_embedding,
                                                                       tar_dim=parameter.model_params.convGRU_params.embedding_filters,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.model_params.split_days or (
                    not w.is_sampling_within_day and parameter.data_params.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=parameter.model_params.split_day_params.filters,
                                                        filter_size=parameter.model_params.split_day_params.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_convGRU.StationaryConvGRU(num_layers=parameter.model_params.convGRU_params.layers,
                                                        in_seq_len=w.samples_per_day,
                                                        in_dim=len(parameter.data_params.features),
                                                        out_seq_len=label_width,
                                                        out_dim=len(parameter.data_params.target),
                                                        units=parameter.model_params.convGRU_params.gru_units,
                                                        filters=parameter.model_params.convGRU_params.embedding_filters,
                                                        kernel_size=parameter.model_params.convGRU_params.embedding_kernel_size,
                                                        gen_mode='unistep',
                                                        is_seq_continuous=is_input_continuous_with_output,
                                                        rate=parameter.model_params.convGRU_params.dropout_rate,
                                                        avg_window=parameter.model_params.decompose_params.avg_window)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(
                        filters=parameter.model_params.convGRU_params.embedding_filters,
                        filter_size=parameter.model_params.split_day_params.kernel_size,
                        n_days=n_days,
                        n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_convGRU.StationaryConvGRU(num_layers=parameter.model_params.convGRU_params.layers,
                                                        in_seq_len=input_width,
                                                        in_dim=len(parameter.data_params.features),
                                                        out_seq_len=label_width,
                                                        out_dim=len(parameter.data_params.target),
                                                        units=parameter.model_params.convGRU_params.gru_units,
                                                        filters=parameter.model_params.convGRU_params.embedding_filters,
                                                        kernel_size=parameter.model_params.convGRU_params.embedding_kernel_size,
                                                        gen_mode='unistep',
                                                        is_seq_continuous=is_input_continuous_with_output,
                                                        rate=parameter.model_params.convGRU_params.dropout_rate,
                                                        avg_window=parameter.model_params.decompose_params.avg_window)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.model_params.bypass,
                                                                out_width=label_width,
                                                                order=parameter.model_params.bypass_params.order,
                                                                in_dim=len(parameter.data_params.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)
            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name=model_name)
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name=model_name)
            datamodel_CL, history = ModelTrainer(dataGnerator=w, model=model,
                                                 sample_rate=parameter.data_params.sample_rate, generatorMode="data",
                                                 testEpoch=testEpoch, name=model_name,
                                                 is_shuffle=parameter.data_params.is_using_shuffle)
            if 'val_loss' in history.history and best_perform > min(history.history["val_loss"]):
                best_model = datamodel_CL
                best_perform = min(history.history["val_loss"])
                with open('./plot/{}/history-{}.json'.format(parameter.exp_params.experiment_label, model_name),
                          'w') as f:
                    json.dump(history.history, f, indent='\t')
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by {}...".format(model_name))
        best_model = datamodel_CL if best_model is None else best_model
        metricsDict = w.allPlot(model=[best_model],
                                name=model_name,
                                scaler=dataUtil.labelScaler,
                                save_csv=parameter.exp_params.save_csv,
                                save_plot=parameter.exp_params.save_plot,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM][model_name] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "stationary_transformer" in parameter.exp_params.model_list:
        model_name = "stationary_transformer"
        best_perform, best_perform2 = float('inf'), float('inf')
        best_model, best_model2 = None, None
        log.info("training {} model...".format(model_name))
        for testEpoch in parameter.exp_params.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            token_len = parameter.model_params.transformer_params.token_length
            input_scalar = Input(shape=(input_width, len(parameter.data_params.features)))
            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.model_params.time_embedding,
                                                                       tar_dim=parameter.model_params.transformer_params.d_model,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.model_params.split_days or (
                    not w.is_sampling_within_day and parameter.data_params.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=parameter.model_params.split_day_params.filters,
                                                        filter_size=parameter.model_params.split_day_params.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_transformer.StationaryTransformer(
                    num_layers=parameter.model_params.transformer_params.layers,
                    d_model=parameter.model_params.transformer_params.d_model,
                    num_heads=parameter.model_params.transformer_params.n_heads,
                    dff=parameter.model_params.transformer_params.dff,
                    src_seq_len=w.samples_per_day,
                    tar_seq_len=label_width,
                    src_dim=parameter.model_params.split_day_params.filters,
                    tar_dim=len(parameter.data_params.target),
                    kernel_size=parameter.model_params.transformer_params.embedding_kernel_size,
                    rate=parameter.model_params.transformer_params.dropout_rate,
                    gen_mode="unistep",
                    is_seq_continuous=is_input_continuous_with_output,
                    token_len=0,
                    avg_window=parameter.model_params.decompose_params.avg_window)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(
                        filters=parameter.model_params.convGRU_params.embedding_filters,
                        filter_size=parameter.model_params.split_day_params.kernel_size,
                        n_days=n_days,
                        n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_transformer.StationaryTransformer(
                    num_layers=parameter.model_params.transformer_params.layers,
                    d_model=parameter.model_params.transformer_params.d_model,
                    num_heads=parameter.model_params.transformer_params.n_heads,
                    dff=parameter.model_params.transformer_params.dff,
                    src_seq_len=input_width,
                    tar_seq_len=label_width,
                    src_dim=len(parameter.data_params.features),
                    tar_dim=len(parameter.data_params.target),
                    kernel_size=parameter.model_params.transformer_params.embedding_kernel_size,
                    rate=parameter.model_params.transformer_params.dropout_rate,
                    gen_mode="unistep",
                    is_seq_continuous=is_input_continuous_with_output,
                    token_len=token_len,
                    avg_window=parameter.model_params.decompose_params.avg_window)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.model_params.bypass,
                                                                out_width=label_width,
                                                                order=parameter.model_params.bypass_params.order,
                                                                in_dim=len(parameter.data_params.features),
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
                                       name=model_name)
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name=model_name)
            datamodel_CL, history = ModelTrainer(dataGnerator=w, model=model,
                                                 sample_rate=parameter.data_params.sample_rate, generatorMode="data",
                                                 testEpoch=testEpoch, name=model_name,
                                                 is_shuffle=parameter.data_params.is_using_shuffle)
            if 'val_loss' in history.history and best_perform > min(history.history["val_loss"]):
                best_model = datamodel_CL
                best_perform = min(history.history["val_loss"])
                with open('./plot/{}/history-{}.json'.format(parameter.exp_params.experiment_label, model_name),
                          'w') as f:
                    json.dump(history.history, f, indent='\t')
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by {}...".format(model_name))
        best_model = datamodel_CL if best_model is None else best_model
        metricsDict = w.allPlot(model=[best_model],
                                name=model_name,
                                scaler=dataUtil.labelScaler,
                                save_csv=parameter.exp_params.save_csv,
                                save_plot=parameter.exp_params.save_plot,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM][model_name] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "znorm_convGRU" in parameter.exp_params.model_list:
        model_name = "znorm_convGRU"
        best_perform, best_perform2 = float('inf'), float('inf')
        best_model, best_model2 = None, None
        log.info(model_name)
        for testEpoch in parameter.exp_params.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            input_scalar = Input(shape=(input_width, len(parameter.data_params.features)))

            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.model_params.time_embedding,
                                                                       tar_dim=parameter.model_params.convGRU_params.embedding_filters,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.model_params.split_days or (
                    not w.is_sampling_within_day and parameter.data_params.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=parameter.model_params.split_day_params.filters,
                                                        filter_size=parameter.model_params.split_day_params.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_convGRU.MovingZNormConvGRU(num_layers=parameter.model_params.convGRU_params.layers,
                                                         in_seq_len=w.samples_per_day,
                                                         in_dim=len(parameter.data_params.features),
                                                         out_seq_len=label_width,
                                                         out_dim=len(parameter.data_params.target),
                                                         units=parameter.model_params.convGRU_params.gru_units,
                                                         filters=parameter.model_params.convGRU_params.embedding_filters,
                                                         kernel_size=parameter.model_params.convGRU_params.embedding_kernel_size,
                                                         gen_mode='unistep',
                                                         is_seq_continuous=is_input_continuous_with_output,
                                                         rate=parameter.model_params.convGRU_params.dropout_rate,
                                                         avg_window=parameter.model_params.decompose_params.avg_window)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(
                        filters=parameter.model_params.convGRU_params.embedding_filters,
                        filter_size=parameter.model_params.split_day_params.kernel_size,
                        n_days=n_days,
                        n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_convGRU.MovingZNormConvGRU(num_layers=parameter.model_params.convGRU_params.layers,
                                                         in_seq_len=input_width,
                                                         in_dim=len(parameter.data_params.features),
                                                         out_seq_len=label_width,
                                                         out_dim=len(parameter.data_params.target),
                                                         units=parameter.model_params.convGRU_params.gru_units,
                                                         filters=parameter.model_params.convGRU_params.embedding_filters,
                                                         kernel_size=parameter.model_params.convGRU_params.embedding_kernel_size,
                                                         gen_mode='unistep',
                                                         is_seq_continuous=is_input_continuous_with_output,
                                                         rate=parameter.model_params.convGRU_params.dropout_rate,
                                                         avg_window=parameter.model_params.decompose_params.avg_window)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.model_params.bypass,
                                                                out_width=label_width,
                                                                order=parameter.model_params.bypass_params.order,
                                                                in_dim=len(parameter.data_params.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)
            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name=model_name)
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name=model_name)
            datamodel_CL, history = ModelTrainer(dataGnerator=w, model=model,
                                                 sample_rate=parameter.data_params.sample_rate, generatorMode="data",
                                                 testEpoch=testEpoch, name=model_name,
                                                 is_shuffle=parameter.data_params.is_using_shuffle)
            if 'val_loss' in history.history and best_perform > min(history.history["val_loss"]):
                best_model = datamodel_CL
                best_perform = min(history.history["val_loss"])
                with open('./plot/{}/history-{}.json'.format(parameter.exp_params.experiment_label, model_name),
                          'w') as f:
                    json.dump(history.history, f, indent='\t')
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by {}...".format(model_name))
        best_model = datamodel_CL if best_model is None else best_model
        metricsDict = w.allPlot(model=[best_model],
                                name=model_name,
                                scaler=dataUtil.labelScaler,
                                save_csv=parameter.exp_params.save_csv,
                                save_plot=parameter.exp_params.save_plot,
                                datamode="data")

        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM][model_name] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    if "znorm_transformer" in parameter.exp_params.model_list:
        model_name = "znorm_transformer"
        best_perform, best_perform2 = float('inf'), float('inf')
        best_model, best_model2 = None, None
        log.info("training {} model...".format(model_name))
        for testEpoch in parameter.exp_params.epoch_list:  # 要在model input前就跑回圈才能讓weight不一樣，weight初始的點是在model input的地方
            token_len = parameter.model_params.transformer_params.token_length
            input_scalar = Input(shape=(input_width, len(parameter.data_params.features)))
            time_embedded = time_embedding_factory.TEFac.new_te_module(command=parameter.model_params.time_embedding,
                                                                       tar_dim=parameter.model_params.transformer_params.d_model,
                                                                       seq_structure=(input_width, shift, label_width))
            if time_embedded is not None:
                input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
                time_embedded = time_embedded(input_time)

            is_splitting_days = parameter.model_params.split_days or (
                    not w.is_sampling_within_day and parameter.data_params.between8_17)
            if is_splitting_days:
                n_days = input_width // w.samples_per_day
                scalar_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                    input_scalar)
                scalar_embedded = MultipleDaysConvEmbed(filters=parameter.model_params.split_day_params.filters,
                                                        filter_size=parameter.model_params.split_day_params.kernel_size,
                                                        n_days=n_days,
                                                        n_samples=w.samples_per_day)(scalar_embedded)
                model = model_transformer.MovingZScoreNormTransformer(
                    num_layers=parameter.model_params.transformer_params.layers,
                    d_model=parameter.model_params.transformer_params.d_model,
                    num_heads=parameter.model_params.transformer_params.n_heads,
                    dff=parameter.model_params.transformer_params.dff,
                    src_seq_len=w.samples_per_day,
                    tar_seq_len=label_width,
                    src_dim=parameter.model_params.split_day_params.filters,
                    tar_dim=len(parameter.data_params.target),
                    kernel_size=parameter.model_params.transformer_params.embedding_kernel_size,
                    rate=parameter.model_params.transformer_params.dropout_rate,
                    gen_mode="unistep",
                    is_seq_continuous=is_input_continuous_with_output,
                    token_len=0,
                    avg_window=parameter.model_params.decompose_params.avg_window)
                if time_embedded is not None:
                    input_time_embedded = SplitInputByDay(n_days=n_days, n_samples=w.samples_per_day)(
                        time_embedded[0])
                    input_time_embedded = MultipleDaysConvEmbed(
                        filters=parameter.model_params.convGRU_params.embedding_filters,
                        filter_size=parameter.model_params.split_day_params.kernel_size,
                        n_days=n_days,
                        n_samples=w.samples_per_day)(input_time_embedded)
                    nonlinear = model(scalar_embedded,
                                      time_embedding_tuple=(input_time_embedded, time_embedded[1], time_embedded[2]))
                else:
                    nonlinear = model(scalar_embedded)
            else:
                model = model_transformer.MovingZScoreNormTransformer(
                    num_layers=parameter.model_params.transformer_params.layers,
                    d_model=parameter.model_params.transformer_params.d_model,
                    num_heads=parameter.model_params.transformer_params.n_heads,
                    dff=parameter.model_params.transformer_params.dff,
                    src_seq_len=input_width,
                    tar_seq_len=label_width,
                    src_dim=len(parameter.data_params.features),
                    tar_dim=len(parameter.data_params.target),
                    kernel_size=parameter.model_params.transformer_params.embedding_kernel_size,
                    rate=parameter.model_params.transformer_params.dropout_rate,
                    gen_mode="unistep",
                    is_seq_continuous=is_input_continuous_with_output,
                    token_len=token_len,
                    avg_window=parameter.model_params.decompose_params.avg_window)
                nonlinear = model(input_scalar, time_embedding_tuple=time_embedded)

            linear = bypass_factory.BypassFac.new_bypass_module(command=parameter.model_params.bypass,
                                                                out_width=label_width,
                                                                order=parameter.model_params.bypass_params.order,
                                                                in_dim=len(parameter.data_params.features),
                                                                window_len=input_width,
                                                                is_within_day=w.is_sampling_within_day,
                                                                samples_per_day=w.samples_per_day)
            if linear is not None:
                linear = linear(input_scalar)
                outputs = tf.keras.layers.Add()([linear, nonlinear])
            else:
                outputs = nonlinear

            if time_embedded is not None:
                model = tf.keras.Model(inputs=[input_scalar, input_time], outputs=outputs, name=model_name)
            else:
                model = tf.keras.Model(inputs=[input_scalar], outputs=outputs, name=model_name)
            datamodel_CL, history = ModelTrainer(dataGnerator=w, model=model,
                                                 sample_rate=parameter.data_params.sample_rate, generatorMode="data",
                                                 testEpoch=testEpoch, name=model_name,
                                                 is_shuffle=parameter.data_params.is_using_shuffle)
            if 'val_loss' in history.history and best_perform > min(history.history["val_loss"]):
                best_model = datamodel_CL
                best_perform = min(history.history["val_loss"])
                with open('./plot/{}/history-{}.json'.format(parameter.exp_params.experiment_label, model_name),
                          'w') as f:
                    json.dump(history.history, f, indent='\t')
            print(best_perform)
            log.info("a model ok")

        log.info("predicting SolarIrradiation by {}...".format(model_name))
        best_model = datamodel_CL if best_model is None else best_model
        metricsDict = w.allPlot(model=[best_model],
                                name=model_name,
                                scaler=dataUtil.labelScaler,
                                save_csv=parameter.exp_params.save_csv,
                                save_plot=parameter.exp_params.save_plot,
                                datamode="data")
        for logM in metricsDict:
            if modelMetricsRecorder.get(logM) is None:
                modelMetricsRecorder[logM] = {}
            modelMetricsRecorder[logM][model_name] = metricsDict[logM]
        pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    metrics_path = "plot/{}/{}".format(parameter.exp_params.experiment_label, "all_metric")
    pd.DataFrame(modelMetricsRecorder).to_csv(Path(metrics_path + ".csv"))

    return modelMetricsRecorder


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(run_eagerly=True)
    # tf.data.experimental.enable_debug_mode()

    # parse and assign arguments
    args = args_parse()
    # create exp. dir
    try:
        os.mkdir("./plot/{}".format(parameter.exp_params.experiment_label))
    except:
        pass
    # run core business logic
    result = run()
    # save exp. config
    with open('./plot/{}/config.txt'.format(parameter.exp_params.experiment_label),
              'w') as f:
        f.write(str(parameter.data_params))
        f.write(str(parameter.exp_params))
        f.write(str(parameter.model_params))
