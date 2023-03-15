import argparse
import os

import keras_tuner
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from pyimagesearch import model_AR, time_embedding, model_transformer, parameter, my_metrics, model_convGRU
from pyimagesearch.datautil import DataUtil
from pyimagesearch.windowsGenerator import WindowGenerator

shift = None
label_width = None
is_input_continuous_with_output = None
train_path = os.path.sep.join([parameter.csv_name])
val_path = None
test_path = None
data = DataUtil(train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                normalise=parameter.norm_mode,
                label_col=parameter.target,
                feature_col=parameter.features,
                split_mode=parameter.split_mode,
                month_sep=parameter.test_month)


class HyperTransformer(keras_tuner.HyperModel):
    def __init__(self, fine_tune: bool = False, tune_architecture: bool = False, tune_training: bool = False):
        super().__init__()
        self.fine_tune = fine_tune
        self.tune_architecture = tune_architecture
        self.tune_training = tune_training

    def build(self, hp):
        if self.tune_training:
            hp.Int("batch_size", 16, 256, step=2, sampling="log")
            hp.Boolean('is_shuffle')
            hp.Choice('loss', ['mse', 'mae'])
            hp.Choice('optimizer', ['Adam', 'RMSprop', 'Adagrad', 'Adadelta'])
        else:
            hp.Fixed("batch_size", 16)
            hp.Fixed('is_shuffle', False)
            hp.Fixed('loss', 'mse')
            hp.Fixed('optimizer', 'Adam')
        if self.tune_architecture:
            hp.Boolean('is_using_stationary_module')
            hp.Boolean('is_using_time_embedding')
            hp.Boolean('is_using_LR')
        else:
            hp.Fixed('is_using_stationary_module', True)
            hp.Fixed('is_using_time_embedding', True)
            hp.Fixed('is_using_LR', True)
        if self.fine_tune:
            hp.Int("input_width", 24, 240, 24)
            hp.Int("layers", 1, 5)
            hp.Int("d_model", 8, 512, step=2, sampling="log")
            hp.Int("heads", 1, 8, step=2, sampling="log")
            hp.Int('dff_multiplier', 1, 8, step=2, sampling="log")
            hp.Int('kernel_size', 1, 17, step=2)
            hp.Float('dropout_rate', 0, 0.5, 0.1, default=0.1)
            hp.Boolean('is_using_pooling')
            hp.Int("token_len", 0, 240, 24)
            hp.Int("avg_window", 3, 17, step=2, parent_name='is_using_stationary_module', parent_values=True)
            hp.Int("LR_order", 24, 240, 24, parent_name='is_using_LR', parent_values=True)
        else:
            hp.Fixed("input_width", 168)
            hp.Fixed("layers", 1)
            hp.Fixed("d_model", 32)
            hp.Fixed("heads", 1)
            hp.Fixed('dff_multiplier', 4)
            hp.Fixed('kernel_size', 3)
            hp.Fixed('dropout_rate', 0.1)
            hp.Fixed('is_using_pooling', False)
            hp.Fixed("token_len", 24)
            hp.Fixed("avg_window", 17, parent_name='is_using_stationary_module', parent_values=True)
            hp.Fixed("LR_order", 24, parent_name='is_using_LR', parent_values=True)
        input_width = hp.get("input_width")
        input_scalar = Input(shape=(input_width, len(parameter.features)))
        if hp.get('is_using_stationary_module'):
            model = model_transformer.StationaryTransformer(num_layers=hp.get("layers"),
                                                            d_model=hp.get("d_model"),
                                                            num_heads=hp.get("heads"),
                                                            dff=hp.get("d_model") * hp.get('dff_multiplier'),
                                                            src_seq_len=input_width,
                                                            tar_seq_len=label_width, src_dim=len(parameter.features),
                                                            tar_dim=len(parameter.target),
                                                            kernel_size=hp.get('kernel_size'),
                                                            rate=hp.get('dropout_rate'),
                                                            gen_mode="unistep",
                                                            is_seq_continuous=is_input_continuous_with_output,
                                                            is_pooling=hp.get('is_using_pooling'),
                                                            token_len=hp.get("token_len"),
                                                            avg_window=hp.get("avg_window"))
        else:
            model = model_transformer.Transformer(num_layers=hp.get("layers"),
                                                  d_model=hp.get("d_model"),
                                                  num_heads=hp.get("heads"),
                                                  dff=hp.get("d_model") * hp.get('dff_multiplier'),
                                                  src_seq_len=input_width,
                                                  tar_seq_len=label_width, src_dim=len(parameter.features),
                                                  tar_dim=len(parameter.target),
                                                  kernel_size=hp.get('kernel_size'),
                                                  rate=hp.get('dropout_rate'),
                                                  gen_mode="unistep",
                                                  is_seq_continuous=is_input_continuous_with_output,
                                                  is_pooling=hp.get('is_using_pooling'),
                                                  token_len=hp.get("token_len"))
        input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
        if hp.get('is_using_time_embedding'):
            embedding = time_embedding.TimeEmbedding(output_dims=hp.get("d_model"),
                                                     input_len=input_width,
                                                     shift_len=shift,
                                                     label_len=label_width)(input_time)
        else:
            embedding = None
        nonlinear = model(input_scalar, time_embedding_tuple=embedding)
        if hp.get('is_using_LR'):
            linear = model_AR.TemporalChannelIndependentLR(hp.get("LR_order"),
                                                           label_width,
                                                           len(parameter.features))(input_scalar)
            outputs = tf.keras.layers.Add()([linear, nonlinear])
        else:
            outputs = nonlinear
        model = Model(inputs=[input_scalar, input_time], outputs=outputs)
        model.compile(loss=hp.get('loss'),
                      optimizer=hp.get('optimizer'),
                      metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsolutePercentageError(),
                               my_metrics.VWMAPE, my_metrics.root_relative_squared_error, my_metrics.corr])
        model.summary()
        return model

    def fit(self, hp, model, **kwargs):
        input_width = hp.get("input_width")
        w = WindowGenerator(input_width=input_width,
                            image_input_width=0,
                            label_width=label_width,
                            shift=shift,

                            trainImages=data.trainImages,
                            trainData=data.train_df[data.feature_col],
                            trainCloud=data.train_df_cloud,
                            trainAverage=data.train_df_average,
                            trainY=data.train_df[data.label_col],

                            valImage=data.valImages,
                            valData=data.val_df[data.feature_col],
                            valCloud=data.val_df_cloud,
                            valAverage=data.val_df_average,
                            valY=data.val_df[data.label_col],

                            testImage=data.testImages,
                            testData=data.test_df[data.feature_col],
                            testCloud=data.test_df_cloud,
                            testAverage=data.test_df_average,
                            testY=data.test_df[data.label_col],

                            batch_size=hp.get("batch_size"),
                            label_columns="ShortWaveDown",
                            samples_per_day=data.samples_per_day,
                            using_timestamp_data=True,
                            using_shuffle=hp.get('is_shuffle'))
        return model.fit(w.trainData(addcloud=parameter.addAverage),
                         validation_data=w.valData(addcloud=parameter.addAverage),
                         epochs=parameter.epochs,
                         **kwargs)


class HyperConvGRU(keras_tuner.HyperModel):
    def __init__(self, fine_tune: bool = False, tune_architecture: bool = False, tune_training: bool = False):
        super().__init__()
        self.fine_tune = fine_tune
        self.tune_architecture = tune_architecture
        self.tune_training = tune_training

    def build(self, hp):
        if self.tune_training:
            hp.Int("batch_size", 16, 256, step=2, sampling="log")
            hp.Boolean('is_shuffle')
            hp.Choice('loss', ['mse', 'mae'])
            hp.Choice('optimizer', ['Adam', 'RMSprop', 'Adagrad', 'Adadelta'])
        else:
            hp.Fixed("batch_size", 16)
            hp.Fixed('is_shuffle', False)
            hp.Fixed('loss', 'mse')
            hp.Fixed('optimizer', 'Adam')
        if self.tune_architecture:
            hp.Boolean('is_using_stationary_module')
            hp.Boolean('is_using_time_embedding')
            hp.Boolean('is_using_LR')
        else:
            hp.Fixed('is_using_stationary_module', True)
            hp.Fixed('is_using_time_embedding', True)
            hp.Fixed('is_using_LR', True)
        if self.fine_tune:
            hp.Int("input_width", 24, 240, 24)
            hp.Int("layers", 1, 5)
            hp.Int('units', 1, 512, step=2, sampling="log")
            hp.Int('filters', 1, 512, step=2, sampling="log")
            hp.Int('kernel_size', 1, 17, step=2)
            hp.Float('dropout_rate', 0, 0.5, 0.1, default=0.1)
            hp.Int("avg_window", 3, 17, step=2, parent_name='is_using_stationary_module', parent_values=True)
            hp.Int("LR_order", 24, 240, 24, parent_name='is_using_LR', parent_values=True)
        else:
            hp.Fixed("input_width", 168)
            hp.Fixed("layers", 1)
            hp.Fixed('units', 32)
            hp.Fixed('filters', 32)
            hp.Fixed('kernel_size', 3)
            hp.Fixed('dropout_rate', 0.1)
            hp.Fixed("avg_window", 17, parent_name='is_using_stationary_module', parent_values=True)
            hp.Fixed("LR_order", 24, parent_name='is_using_LR', parent_values=True)
        input_width = hp.get("input_width")
        input_scalar = Input(shape=(input_width, len(parameter.features)))
        if hp.get('is_using_stationary_module'):
            model = model_convGRU.StationaryConvGRU(num_layers=hp.get("layers"),
                                                    in_seq_len=input_width,
                                                    in_dim=len(parameter.features),
                                                    out_seq_len=label_width,
                                                    out_dim=len(parameter.target),
                                                    units=hp.get('units'),
                                                    filters=hp.get('filters'),
                                                    kernel_size=hp.get('kernel_size'),
                                                    gen_mode='unistep',
                                                    is_seq_continuous=is_input_continuous_with_output,
                                                    rate=hp.get('dropout_rate'),
                                                    avg_window=hp.get("avg_window"))
        else:
            model = model_convGRU.ConvGRU(num_layers=hp.get("layers"),
                                          in_seq_len=input_width,
                                          in_dim=len(parameter.features),
                                          out_seq_len=label_width,
                                          out_dim=len(parameter.target),
                                          units=hp.get('units'),
                                          filters=hp.get('filters'),
                                          kernel_size=hp.get('kernel_size'),
                                          gen_mode='unistep',
                                          is_seq_continuous=is_input_continuous_with_output,
                                          rate=hp.get('dropout_rate'))
        input_time = Input(shape=(input_width + shift + label_width, len(time_embedding.vocab_size)))
        if hp.get('is_using_time_embedding'):
            embedding = time_embedding.TimeEmbedding(output_dims=hp.get('filters'),
                                                     input_len=input_width,
                                                     shift_len=shift,
                                                     label_len=label_width)(input_time)
        else:
            embedding = None
        nonlinear = model(input_scalar, time_embedding_tuple=embedding)
        if hp.get('is_using_LR'):
            linear = model_AR.TemporalChannelIndependentLR(hp.get("LR_order"),
                                                           label_width,
                                                           len(parameter.features))(input_scalar)
            outputs = tf.keras.layers.Add()([linear, nonlinear])
        else:
            outputs = nonlinear
        model = Model(inputs=[input_scalar, input_time], outputs=outputs)
        model.compile(loss=hp.get('loss'),
                      optimizer=hp.get('optimizer'),
                      metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsolutePercentageError(),
                               my_metrics.VWMAPE, my_metrics.root_relative_squared_error, my_metrics.corr])
        model.summary()
        return model

    def fit(self, hp, model, **kwargs):
        input_width = hp.get("input_width")
        w = WindowGenerator(input_width=input_width,
                            image_input_width=0,
                            label_width=label_width,
                            shift=shift,

                            trainImages=data.trainImages,
                            trainData=data.train_df[data.feature_col],
                            trainCloud=data.train_df_cloud,
                            trainAverage=data.train_df_average,
                            trainY=data.train_df[data.label_col],

                            valImage=data.valImages,
                            valData=data.val_df[data.feature_col],
                            valCloud=data.val_df_cloud,
                            valAverage=data.val_df_average,
                            valY=data.val_df[data.label_col],

                            testImage=data.testImages,
                            testData=data.test_df[data.feature_col],
                            testCloud=data.test_df_cloud,
                            testAverage=data.test_df_average,
                            testY=data.test_df[data.label_col],

                            batch_size=hp.get("batch_size"),
                            label_columns="ShortWaveDown",
                            samples_per_day=data.samples_per_day,
                            using_timestamp_data=True,
                            using_shuffle=hp.get('is_shuffle'))
        return model.fit(w.trainData(addcloud=parameter.addAverage),
                         validation_data=w.valData(addcloud=parameter.addAverage),
                         epochs=parameter.epochs,
                         **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hyper-parameters tuning')
    parser.add_argument('-s', '--shift', type=int, default=parameter.shifted_width,
                        help='lag between input and output sequence')
    parser.add_argument('-o', '--output', type=int, default=parameter.label_width, help='length of output sequence')
    parser.add_argument('-t', '--trials', type=int, default=3, help='max trials when perform hp searching')
    parser.add_argument('-ft', '--finetune', default=False, action='store_true', help='fine-tuning or not')
    parser.add_argument('-at', '--arch', default=False, action='store_true', help='tuning network architecture or not')
    parser.add_argument('-tt', '--training', default=False, action='store_true', help='tuning training parameters or not')
    parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite tuning history or not')
    args = parser.parse_args()

    shift = args.shift
    label_width = args.output
    is_input_continuous_with_output = (shift == 0) and (not parameter.between8_17)

    exp_setup_str = "s{}o{}_{}".format(shift, label_width, parameter.csv_name)
    tuning_setup_str = 'ft{}_at{}_tt{}'.format(int(args.finetune), int(args.arch), int(args.training))
    tuner1 = keras_tuner.BayesianOptimization(
        HyperTransformer(fine_tune=args.finetune, tune_architecture=args.arch, tune_training=args.training),
        objective="val_loss",
        max_trials=args.trials,
        overwrite=args.overwrite,
        directory="tuning/{}".format(exp_setup_str),
        project_name="{}/transformer".format(tuning_setup_str)
    )
    tuner1.search(callbacks=[parameter.earlystoper])

    tuner2 = keras_tuner.BayesianOptimization(
        HyperConvGRU(fine_tune=args.finetune, tune_architecture=args.arch, tune_training=args.training),
        objective="val_loss",
        max_trials=args.trials,
        overwrite=args.overwrite,
        directory="tuning/{}".format(exp_setup_str),
        project_name="{}/convGRU".format(tuning_setup_str)
    )
    tuner2.search(callbacks=[parameter.earlystoper])
    print("\n\n#####################Transformer's Best Result#####################")
    tuner1.results_summary(num_trials=1)
    print("\n\n#####################ConvGRU's Best Result#####################")
    tuner2.results_summary(num_trials=1)