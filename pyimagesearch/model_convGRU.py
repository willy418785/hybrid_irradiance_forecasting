import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Concatenate, Dropout, Dense, Flatten, \
    Input, Add
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, LayerNormalization, Reshape, ZeroPadding2D, \
    MultiHeadAttention
from tensorflow.keras.models import Model

from pyimagesearch import parameter
from pyimagesearch.datautil import DataUtil
from pyimagesearch.windowsGenerator import WindowGenerator

gen_modes = ['unistep', 'auto']


class Encoder(tf.keras.layers.Layer):
    def __init__(self, layers, units, filters=None, rate=0.1):
        super().__init__()
        self.layers = layers
        self.units = units
        if filters is None:
            self.filters = units
        else:
            self.filters = filters

        self.conv = Conv1D(filters=filters, kernel_size=5, strides=1, padding="same", activation='elu')

        self.gru_layers = [
            GRU(units, return_sequences=True, return_state=True, dropout=rate)
            for _ in range(layers)]

    def call(self, input_seq, training):
        x = self.conv(input_seq)
        states = []
        for i in range(self.layers):
            x, state = self.gru_layers[i](x, training=training)
            states.append(state)
        states = tf.stack(states, axis=-1)
        return x, states


class Decoder(tf.keras.layers.Layer):
    def __init__(self, layers, units, filters=None, rate=0.1):
        super().__init__()
        self.layers = layers
        self.units = units
        if filters is None:
            self.filters = units
        else:
            self.filters = filters
        self.gru_layers = [
            GRU(units, return_sequences=True, return_state=True, dropout=rate)
            for _ in range(layers)]

    def call(self, input_seq, initial_states, training):
        x = input_seq
        states = []
        for i in range(self.layers):
            x, state = self.gru_layers[i](x, training=training, initial_state=initial_states[:, :, i])
            states.append(state)
        states = tf.stack(states, axis=-1)
        return x, states


class ConvGRU(tf.keras.Model):
    def __init__(self, num_layers, in_seq_len, in_dim, out_seq_len, out_dim, units, filters=None, gen_mode='unistep',
                 is_seq_continuous=False, rate=0.1):
        super().__init__()
        assert gen_mode in gen_modes
        self.num_layers = num_layers
        self.out_seq_len = out_seq_len
        self.out_dim = out_dim
        self.units = units
        self.gen_mode = gen_mode
        self.is_seq_continuous = is_seq_continuous
        if filters is None:
            self.filters = units
        else:
            self.filters = filters

        self.encoder = Encoder(num_layers, units, filters, rate=rate)
        self.decoder = Decoder(num_layers, units, filters, rate=rate)
        if gen_mode == "unistep":
            self.fc = Sequential([Dense(out_seq_len * out_dim), Reshape((out_seq_len, out_dim))])
        elif gen_mode == 'auto':
            self.fc = Dense(out_dim)
        self.build(input_shape=(None, in_seq_len, in_dim))

    def call(self, input_seq, training):
        enc_seq, states = self.encoder(input_seq, training)
        if self.gen_mode == "unistep":
            states = Flatten()(states)
            output = self.fc(states)
        elif self.gen_mode == 'auto':
            @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1, self.units), dtype=tf.float32),
                                          tf.TensorSpec(shape=(None, self.units, self.num_layers), dtype=tf.float32)],
                         experimental_relax_shapes=True)
            def autoregress(inputs, states):
                # print('[Side Effect] Retracing graph')
                out = tf.stack([tf.shape(inputs)[0], 0, self.out_dim])
                out = tf.fill(out, 0.0)
                for i in tf.range(self.out_seq_len):
                    tf.autograph.experimental.set_loop_options(
                        shape_invariants=[(out, tf.TensorShape([None, None, self.out_dim]))])
                    inputs, states = self.decoder(inputs, states, training)
                    tmp = self.fc(inputs)
                    out = tf.concat([out, tmp[:, -1:, :]], axis=1)
                return out

            if self.is_seq_continuous:
                inputs = enc_seq[:, -1:, :]
            else:
                inputs = tf.stack([tf.shape(enc_seq)[0], 1, self.units])
                inputs = tf.fill(inputs, 0.0)
            output = autoregress(inputs, states)
        return output


if __name__ == '__main__':
    train_path_with_weather_info = os.path.sep.join(["../2020weatherInfoNormalized.csv"])
    data_with_weather_info = DataUtil(train_path=train_path_with_weather_info,
                                      val_path=None,
                                      test_path=None,
                                      normalise=0,
                                      label_col=parameter.target,
                                      feature_col=parameter.features,
                                      split_mode=parameter.split_mode,
                                      month_sep=parameter.test_month)
    dataUtil = data_with_weather_info
    w2 = WindowGenerator(input_width=10,
                         image_input_width=0,
                         label_width=10,
                         shift=parameter.after_minutes,

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
                         label_columns="ShortWaveDown")
    model = ConvGRU(num_layers=1, in_seq_len=10, in_dim=len(parameter.features), out_seq_len=10, out_dim=len(parameter.target), units=5, filters=100,
                    gen_mode='auto',
                    is_seq_continuous=True)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                  , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()])
    model.summary()
    tf.keras.backend.clear_session()
    # history = model.fit(w2.trainData(addcloud=parameter.addAverage),
    #                     validation_data=w2.valData(addcloud=parameter.addAverage),
    #                     epochs=100, batch_size=5, callbacks=[parameter.earlystoper])
    for x, y in w2.trainData(addcloud=parameter.addAverage):
        c = model(x[:1, :, :])
    model.summary()
    pass
