import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from pyimagesearch import parameter
from pyimagesearch.datautil import DataUtil
from pyimagesearch.windowsGenerator import WindowGenerator
gen_modes = ['unistep', 'auto', "mlp"]

class Config():
    order = 24


class AR(tf.keras.layers.Layer):
    def __init__(self, order, tar_seq_len, src_dims):
        super().__init__()
        self.order = order
        self.tar_seq_len = tar_seq_len
        self.src_dims = src_dims
        self.linear = Dense(1)

    def call(self, inputs, training):
        temporal_last = tf.transpose(inputs, perm=[0, 2, 1])
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, self.src_dims, self.order), dtype=tf.float32)],
                     experimental_relax_shapes=True)
        def autoregress(tar):
            out = tf.stack([tf.shape(inputs)[0], self.src_dims, 0])
            out = tf.fill(out, 0.0)
            for i in tf.range(self.tar_seq_len):
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[(out, tf.TensorShape([None, self.src_dims, None]))])
                y = self.linear(tar)
                tar = tf.concat([tar[:, :, 1:], y], axis=-1)
                out = tf.concat([out, y], axis=-1)
            return out
        output = autoregress(temporal_last[:, :, -self.order:])
        output = tf.transpose(output, perm=[0, 2, 1])
        return output

class ChannelIndependentAR(tf.keras.layers.Layer):
    def __init__(self, order, tar_seq_len, src_dims):
        super().__init__()
        self.order = order
        self.tar_seq_len = tar_seq_len
        self.src_dims = src_dims
        self.linears = [Dense(1) for _ in range(src_dims)]

    def call(self, inputs, training):
        temporal_last = tf.transpose(inputs, perm=[0, 2, 1])
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, self.src_dims, self.order), dtype=tf.float32)],
                     experimental_relax_shapes=True)
        def autoregress(tar):
            out = tf.stack([tf.shape(inputs)[0], self.src_dims, 0])
            out = tf.fill(out, 0.0)
            for _ in tf.range(self.tar_seq_len):
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[(out, tf.TensorShape([None, self.src_dims, None]))])
                y = [self.linears[i](tar[:, i, :]) for i in range(self.src_dims)]
                y = tf.stack(y, axis=1)
                tar = tf.concat([tar[:, :, 1:], y], axis=-1)
                out = tf.concat([out, y], axis=-1)
            return out
        output = autoregress(temporal_last[:, :, -self.order:])
        output = tf.transpose(output, perm=[0, 2, 1])
        return output

class TemporalChannelIndependentLR(tf.keras.layers.Layer):
    def __init__(self, order, tar_seq_len, src_dims):
        super().__init__()
        self.order = order
        self.tar_seq_len = tar_seq_len
        self.src_dims = src_dims
        self.linears = [Dense(self.tar_seq_len) for _ in range(src_dims)]

    def call(self, inputs, training):
        temporal_last = tf.transpose(inputs, perm=[0, 2, 1])
        y = [self.linears[i](temporal_last[:, i, -self.order:]) for i in range(self.src_dims)]
        output = tf.stack(y, axis=1)
        output = tf.transpose(output, perm=[0, 2, 1])
        return output

if __name__ == '__main__':
    train_path_with_weather_info = os.path.sep.join(["../{}".format(parameter.data_params.csv_name)])
    data_with_weather_info = DataUtil(train_path=train_path_with_weather_info,
                                      val_path=None,
                                      test_path=None,
                                      normalise=0,
                                      label_col=parameter.data_params.target,
                                      feature_col=parameter.data_params.features,
                                      split_mode=parameter.data_params.split_mode,
                                      month_sep=parameter.data_params.test_month)
    dataUtil = data_with_weather_info
    src_len = 10
    tar_len = 10
    shift = 0
    w2 = WindowGenerator(input_width=src_len,
                         image_input_width=0,
                         label_width=tar_len,
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

                         batch_size=1,
                         label_columns="ShortWaveDown",
                         samples_per_day=dataUtil.samples_per_day)
    model = Sequential([Input(shape=(src_len, len(parameter.data_params.features))), TemporalChannelIndependentLR(5, 10, len(parameter.data_params.features))])
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                  , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()])
    model.summary()
    tf.keras.backend.clear_session()
    # history = model.fit(w2.trainData(addcloud=parameter.addAverage),
    #                     validation_data=w2.valData(addcloud=parameter.addAverage),
    #                     epochs=100, batch_size=5, callbacks=[parameter.earlystoper])
    for x, y in w2.train(w2.samples_per_day, addcloud=False, using_timestamp_data=False, is_shuffle=False):
        c = model(x[:1, :, :])
    model.summary()
    pass
