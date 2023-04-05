import argparse
import os

import keras_tuner
import tensorflow as tf

from pyimagesearch import my_metrics, parameter
from pyimagesearch.datautil import DataUtil
from pyimagesearch.windowsGenerator import WindowGenerator


class Persistence(tf.keras.Model):
    def __init__(self, is_within_day, samples_per_day, output_width):
        super().__init__()
        self.is_within_day = is_within_day
        self.samples_per_day = samples_per_day
        self.output_width = output_width

    def call(self, inputs):
        if self.is_within_day:
            result = inputs[:, -1:, :]
            result = tf.repeat(result, self.output_width, axis=1)
        else:
            result = inputs[:, self.samples_per_day * -1:, :]
            result = tf.tile(result, [1, int(self.output_width / self.samples_per_day), 1])
        return result


class MA(tf.keras.Model):
    def __init__(self, window_len, is_within_day, samples_per_day, output_width):
        super().__init__()
        self.window_len = window_len
        self.is_within_day = is_within_day
        self.samples_per_day = samples_per_day
        self.output_width = output_width

    def call(self, inputs):
        assert inputs.shape[1] >= self.window_len
        inputs = inputs[:, -self.window_len:, :]
        if self.is_within_day:
            result = tf.reduce_mean(inputs[:, -self.window_len:, :], axis=1, keepdims=True)
            result = tf.repeat(result, self.output_width, axis=1)
        else:
            assert self.window_len % self.samples_per_day == 0
            assert self.output_width % self.samples_per_day == 0
            result = tf.concat([tf.reduce_mean(inputs[:, i::self.samples_per_day, :], axis=1, keepdims=True)
                                for i in range(self.samples_per_day)],
                               axis=1)
            result = tf.tile(result, [1, int(self.output_width / self.samples_per_day), 1])
        return result


if __name__ == '__main__':
    train_path = os.path.sep.join(['../', parameter.data_params.csv_name])
    val_path = None
    test_path = None
    data = DataUtil(train_path=train_path,
                    val_path=val_path,
                    test_path=test_path,
                    normalise=parameter.data_params.norm_mode,
                    label_col=parameter.data_params.target,
                    feature_col=parameter.data_params.target,
                    split_mode=parameter.data_params.split_mode,
                    month_sep=parameter.data_params.test_month,
                    val_split=0.01,
                    test_split=0.01)
    data.test_df = data.train_df
    parser = argparse.ArgumentParser(description='hyper-parameters tuning')
    parser.add_argument('-s', '--shift', type=int, default=parameter.data_params.shifted_width,
                        help='lag between input and output sequence')
    parser.add_argument('-o', '--output', type=int, default=parameter.data_params.label_width,
                        help='length of output sequence')
    args = parser.parse_args()
    WMAPE_dict = {}
    corr_dict = {}
    for ma_width in range(24, 721, 24):
        w_for_MA = WindowGenerator(input_width=ma_width,
                                   image_input_width=0,
                                   label_width=args.output,
                                   shift=args.shift,

                                   trainImages=data.trainImages,
                                   trainData=data.train_df[data.feature_col],
                                   trainCloud=data.train_df_cloud,  ######
                                   trainAverage=data.train_df_average,  ######
                                   trainY=data.train_df[data.label_col],

                                   valImage=data.valImages,
                                   valData=data.val_df[data.feature_col],
                                   valCloud=data.val_df_cloud,  ######
                                   valAverage=data.val_df_average,  ######
                                   valY=data.val_df[data.label_col],

                                   testImage=data.testImages,
                                   testData=data.test_df[data.feature_col],
                                   testCloud=data.test_df_cloud,  ######
                                   testAverage=data.test_df_average,  ######
                                   testY=data.test_df[data.label_col],

                                   batch_size=parameter.exp_params.batch_size,
                                   label_columns="ShortWaveDown",
                                   samples_per_day=data.samples_per_day)
        movingAverage = MA(ma_width, w_for_MA.is_sampling_within_day, w_for_MA.samples_per_day, args.output)
        movingAverage.compile(loss=tf.losses.MeanSquaredError(),
                              metrics=[tf.metrics.MeanAbsoluteError(),
                                       tf.metrics.MeanAbsolutePercentageError(),
                                       my_metrics.VWMAPE,
                                       my_metrics.corr])
        metricsDict = w_for_MA.allPlot(model=[movingAverage],
                                       name="MA",
                                       scaler=data.labelScaler,
                                       datamode="data",
                                       save_plot=False,
                                       save_csv=False)

        WMAPE_dict[ma_width] = metricsDict['WMAPE']
        corr_dict[ma_width] = metricsDict['corr']
        print(WMAPE_dict)
        print(corr_dict)
    w_for_persistance = WindowGenerator(input_width=args.output,
                                        image_input_width=0,
                                        label_width=args.output,
                                        shift=args.shift,

                                        trainImages=data.trainImages,
                                        trainData=data.train_df[data.feature_col],
                                        trainCloud=data.train_df_cloud,  ######
                                        trainAverage=data.train_df_average,  ######
                                        trainY=data.train_df[data.label_col],

                                        valImage=data.valImages,
                                        valData=data.val_df[data.feature_col],
                                        valCloud=data.val_df_cloud,  ######
                                        valAverage=data.val_df_average,  ######
                                        valY=data.val_df[data.label_col],

                                        testImage=data.testImages,
                                        testData=data.test_df[data.feature_col],
                                        testCloud=data.test_df_cloud,  ######
                                        testAverage=data.test_df_average,  ######
                                        testY=data.test_df[data.label_col],

                                        batch_size=parameter.exp_params.batch_size,
                                        label_columns="ShortWaveDown",
                                        samples_per_day=data.samples_per_day)
    baseline = Persistence(w_for_persistance.is_sampling_within_day,
                           w_for_persistance.samples_per_day,
                           args.output)
    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()
                         , tf.metrics.MeanAbsolutePercentageError()
                         , my_metrics.VWMAPE
                         , my_metrics.corr])
    persistence_metrics = w_for_persistance.allPlot(model=[baseline],
                                                    name="Persistence",
                                                    scaler=data.labelScaler,
                                                    datamode="data",
                                                    save_plot=False,
                                                    save_csv=False)

    print('#####################################################')
    best_for_WMAPE = min(WMAPE_dict, key=WMAPE_dict.get)
    best_for_corr = max(corr_dict, key=corr_dict.get)
    print("          Dataset: {}".format(train_path))
    print('        Shift len: {}'.format(args.shift))
    print('       Output len: {}'.format(args.output))
    print("       Best WMAPE: {} with MA width of {}".format(WMAPE_dict[best_for_WMAPE], best_for_WMAPE))
    print("       Best  CORR: {} with MA width of {}".format(corr_dict[best_for_corr], best_for_corr))
    print("Persistence WMAPE: {}".format(persistence_metrics['WMAPE']))
    print("Persistence  CORR: {}".format(persistence_metrics['corr']))
    #
    #
    #

# #####################################################
#           Dataset: ../\EC.csv
#         Shift len: 0
#        Output len: 24
#        Best WMAPE: 0.07847143709659576 with MA width of 24
#        Best  CORR: 0.9080327749252319 with MA width of 192
# Persistence WMAPE: 0.07847143709659576
# Persistence  CORR: 0.8894655704498291
# #####################################################
#           Dataset: ../\EC.csv
#         Shift len: 0
#        Output len: 96
#        Best WMAPE: 0.08754201233386993 with MA width of 168
#        Best  CORR: 0.8986331224441528 with MA width of 168
# Persistence WMAPE: 0.09951511025428772
# Persistence  CORR: 0.8456669449806213
# #####################################################
#           Dataset: ../\EC.csv
#         Shift len: 0
#        Output len: 168
#        Best WMAPE: 0.09105895459651947 with MA width of 168
#        Best  CORR: 0.8920783996582031 with MA width of 168
# Persistence WMAPE: 0.1048528179526329
# Persistence  CORR: 0.8467888832092285
# #####################################################
#           Dataset: ../\EC.csv
#         Shift len: 0
#        Output len: 336
#        Best WMAPE: 0.0991450846195221 with MA width of 144
#        Best  CORR: 0.8782457709312439 with MA width of 168
# Persistence WMAPE: 0.11511020362377167
# Persistence  CORR: 0.8334034085273743


# #####################################################
#           Dataset: ../\dataset_renheo_[2019].csv
#         Shift len: 0
#        Output len: 24
#        Best WMAPE: 0.42745229601860046 with MA width of 672
#        Best  CORR: 0.8492820262908936 with MA width of 696
# Persistence WMAPE: 0.478567898273468
# Persistence  CORR: 0.7851826548576355
#####################################################
#           Dataset: ../\dataset_renheo_[2019].csv
#         Shift len: 0
#        Output len: 96
#        Best WMAPE: 0.4260925352573395 with MA width of 696
#        Best  CORR: 0.8487889766693115 with MA width of 696
# Persistence WMAPE: 0.5407254099845886
# Persistence  CORR: 0.7366670370101929
#####################################################
#           Dataset: ../\dataset_renheo_[2019].csv
#         Shift len: 0
#        Output len: 168
#        Best WMAPE: 0.4261748492717743 with MA width of 696
#        Best  CORR: 0.8475924730300903 with MA width of 528
# Persistence WMAPE: 0.5703415870666504
# Persistence  CORR: 0.7234576940536499
#####################################################
#           Dataset: ../\dataset_renheo_[2019].csv
#         Shift len: 0
#        Output len: 336
#        Best WMAPE: 0.4236150085926056 with MA width of 648
#        Best  CORR: 0.8487817049026489 with MA width of 696
# Persistence WMAPE: 0.5532902479171753
# Persistence  CORR: 0.7455820441246033


# #####################################################
#           Dataset: ../\2020final.csv
#         Shift len: 0
#        Output len: 24
#        Best WMAPE: 0.48449480533599854 with MA width of 120
#        Best  CORR: 0.8245481252670288 with MA width of 720
# Persistence WMAPE: 0.48463645577430725
# Persistence  CORR: 0.7828278541564941
# #####################################################
#           Dataset: ../\2020final.csv
#         Shift len: 0
#        Output len: 96
#        Best WMAPE: 0.49338969588279724 with MA width of 672
#        Best  CORR: 0.8198636174201965 with MA width of 672
# Persistence WMAPE: 0.5685784816741943
# Persistence  CORR: 0.7204545140266418
# #####################################################
#           Dataset: ../\2020final.csv
#         Shift len: 0
#        Output len: 168
#        Best WMAPE: 0.49925705790519714 with MA width of 720
#        Best  CORR: 0.816164493560791 with MA width of 720
# Persistence WMAPE: 0.5709657669067383
# Persistence  CORR: 0.7242510914802551
# #####################################################
#           Dataset: ../\2020final.csv
#         Shift len: 0
#        Output len: 336
#        Best WMAPE: 0.5042690634727478 with MA width of 720
#        Best  CORR: 0.8111436367034912 with MA width of 720
# Persistence WMAPE: 0.5987861752510071
# Persistence  CORR: 0.7046056389808655
