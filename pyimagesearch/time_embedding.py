import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Concatenate, Dropout, Dense, Flatten, \
    Input, Add
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, LayerNormalization, Reshape, ZeroPadding2D, \
    MultiHeadAttention, Embedding
from tensorflow.keras.models import Model

from pyimagesearch import parameter, model_AR
from pyimagesearch.datautil import DataUtil

import tensorflow.keras.backend as K

vocab_size = {'month': 12, 'day': 31, "hour": 24, 'minute': 60}


class Config():
    pass


class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, output_dims, input_len, shift_len, label_len):
        super().__init__()
        self.output_dims = output_dims
        self.input_len = input_len
        self.shift_len = shift_len
        self.label_len = label_len
        self.input_slice = slice(None, input_len, None)
        self.shift_slice = slice(input_len, input_len + shift_len, None)
        self.label_slice = slice(-label_len, None, None)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.output_dims),
                                 initializer='uniform',
                                 trainable=True)
        self.P = self.add_weight(name='P',
                                 shape=(1, self.output_dims),
                                 initializer='uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        linear = K.dot(inputs, self.W) + self.P
        output = K.concatenate([linear[:, :, 0:1], K.sin(linear[:, :, 1:])], -1)
        return output[:, self.input_slice, :], output[:, self.shift_slice, :], output[:, self.label_slice, :]


class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, output_dims, input_len, shift_len, label_len):
        super().__init__()
        self.output_dims = output_dims
        self.input_len = input_len
        self.shift_len = shift_len
        self.label_len = label_len
        self.embeds = [Embedding(input_dim=vocab_size[key], output_dim=output_dims,
                                 input_length=self.input_len + self.shift_len + self.label_len,
                                 name="{}_embedding".format(key)) for key in vocab_size]
        self.input_slice = slice(None, input_len, None)
        self.shift_slice = slice(input_len, input_len + shift_len, None)
        self.label_slice = slice(-label_len, None, None)

    def call(self, inputs):
        output = tf.stack([tf.shape(inputs)[0], self.input_len + self.shift_len + self.label_len, self.output_dims])
        output = tf.fill(output, 0.0)
        for i, embed in enumerate(self.embeds):
            output += embed(inputs[:, :, i])
        return output[:, self.input_slice, :], output[:, self.shift_slice, :], output[:, self.label_slice, :]


class SinCosTimeEncoding(tf.keras.layers.Layer):
    def __init__(self, output_dims, input_len, shift_len, label_len):
        super().__init__()
        self.output_dims = output_dims
        self.input_len = input_len
        self.shift_len = shift_len
        self.label_len = label_len
        self.input_slice = slice(None, input_len, None)
        self.shift_slice = slice(input_len, input_len + shift_len, None)
        self.label_slice = slice(-label_len, None, None)
        self.linear = Dense(self.output_dims)

    def call(self, inputs):
        output = []
        for i, key in enumerate(list(vocab_size)):
            sin_transformed = tf.math.sin(2*np.pi*inputs[:, :, i] / vocab_size[key])
            cos_transformed = tf.math.cos(2*np.pi*inputs[:, :, i] / vocab_size[key])
            transformed = tf.stack([sin_transformed, cos_transformed], -1)
            output.append(transformed)
        output = tf.concat(output, -1)
        output = self.linear(output)
        return output[:, self.input_slice, :], output[:, self.shift_slice, :], output[:, self.label_slice, :]

if __name__ == '__main__':

    from pyimagesearch.windowsGenerator import WindowGenerator
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
    src_len = 15
    shift = 5
    tar_len = 10
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

                         batch_size=32,
                         label_columns="ShortWaveDown",
                         samples_per_day=dataUtil.samples_per_day)
    input_scalar = Input(shape=(src_len, len(parameter.data_params.features)))
    input_time = Input(shape=(src_len + shift + tar_len, len(vocab_size)))
    LR = model_AR.TemporalChannelIndependentLR(model_AR.Config.order, tar_seq_len=tar_len,
                                               src_dims=len(parameter.data_params.features))(input_scalar)
    embedding = TimeEmbedding(output_dims=32, input_len=src_len, shift_len=shift, label_len=tar_len)(input_time)
    model = Model(inputs=[input_scalar, input_time], outputs=[LR, embedding])
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                  , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()])
    model.summary()
    tf.keras.backend.clear_session()
    # history = model.fit(w2.trainData(addcloud=parameter.addAverage),
    #                     validation_data=w2.valData(addcloud=parameter.addAverage),
    #                     epochs=100, batch_size=5, callbacks=[parameter.earlystoper])
    for x, y in w2.train(parameter.data_params.sample_rate, addcloud=parameter.data_params.addAverage):
        c = model(x)
        pass
    pass
