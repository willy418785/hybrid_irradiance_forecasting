import os

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D, GRU, Reshape

from pyimagesearch import parameter, time_embedding
from pyimagesearch.datautil import DataUtil
from pyimagesearch.series_decomposition import SeriesDecompose
from pyimagesearch.windowsGenerator import WindowGenerator
from pyimagesearch.model_transformer import positional_encoding

gen_modes = ['unistep', 'auto', "mlp"]


class Config():
    layers = 3
    embedding_filters = 32
    gru_units = 16
    embedding_kernel_size = 3
    dropout_rate = 0.1


class Encoder(tf.keras.layers.Layer):
    def __init__(self, layers, units, seq_len=None, filters=None, kernel_size=3, rate=0.1):
        super().__init__()
        self.layers = layers
        self.units = units
        if filters is None:
            self.filters = units
        else:
            self.filters = filters
        self.kernel_size = kernel_size
        self.pos_vec = positional_encoding(seq_len, self.filters) if seq_len is not None else None
        self.conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=1, padding="same",
                           activation='elu')

        self.gru_layers = [
            GRU(units, return_sequences=True, return_state=True, dropout=rate)
            for _ in range(layers)]

    def call(self, input_seq, training, timestamp_embedding=None):
        x = self.conv(input_seq)
        if timestamp_embedding is not None:
            x += timestamp_embedding
        if self.pos_vec is not None:
            x += self.pos_vec
        states = []
        for i in range(self.layers):
            x, state = self.gru_layers[i](x, training=training)
            states.append(state)
        states = tf.stack(states, axis=-1)
        return x, states


class Decoder(tf.keras.layers.Layer):
    def __init__(self, layers, units, seq_len=None, filters=None, kernel_size=3, rate=0.1):
        super().__init__()
        self.layers = layers
        self.units = units
        if filters is None:
            self.filters = units
        else:
            self.filters = filters
        self.kernel_size = kernel_size
        self.pos_vec = positional_encoding(seq_len, self.filters) if seq_len is not None else None

        self.conv = Conv1D(filters=self.filters, kernel_size=kernel_size, strides=1, padding="causal",
                           activation='elu')
        self.gru_layers = [
            GRU(units, return_sequences=True, return_state=True, dropout=rate)
            for _ in range(layers)]

    def call(self, input_seq, initial_states, training, timestamp_embedding=None, iter_step=None):
        x = self.conv(input_seq)
        states = []
        if timestamp_embedding is not None:
            x += timestamp_embedding if iter_step is None else timestamp_embedding[:, iter_step:iter_step + 1, :]
        if self.pos_vec is not None:
            x += self.pos_vec if iter_step is None else self.pos_vec[:, iter_step:iter_step + 1, :]
        for i in range(self.layers):
            x, state = self.gru_layers[i](x, training=training, initial_state=initial_states[:, :, i])
            states.append(state)
        states = tf.stack(states, axis=-1)
        return x, states


class ConvGRU(tf.keras.Model):
    def __init__(self, num_layers, in_seq_len, in_dim, out_seq_len, out_dim, units, filters=None, kernel_size=3,
                 gen_mode='unistep',
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

        self.encoder = Encoder(num_layers, units, seq_len=in_seq_len, filters=filters, kernel_size=kernel_size,
                               rate=rate)
        self.decoder = Decoder(num_layers, units, seq_len=out_seq_len, filters=filters, kernel_size=kernel_size,
                               rate=rate)
        if gen_mode == "unistep":
            self.fc = Dense(out_dim)
        elif gen_mode == 'auto':
            self.fc = Dense(out_dim)
        elif gen_mode == "mlp":
            self.fc = Sequential([Dense(out_seq_len * out_dim), Reshape((out_seq_len, out_dim))])
        # self.build(input_shape=(None, in_seq_len, in_dim))

    def call(self, input_seq, training, time_embedding_tuple=None):
        src_timestamps, shift_timestamps, tar_timestamps = time_embedding_tuple if time_embedding_tuple is not None else (
            None, None, None)
        enc_seq, states = self.encoder(input_seq, training=training, timestamp_embedding=src_timestamps)
        if self.gen_mode == "unistep":
            inputs = tf.stack([tf.shape(enc_seq)[0], self.out_seq_len, self.filters])
            inputs = tf.fill(inputs, 0.0)
            dec_out, states = self.decoder(inputs, states, training=training, timestamp_embedding=tar_timestamps)
            output = self.fc(dec_out)
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
                        shape_invariants=[(out, tf.TensorShape([None, None, self.out_dim]))
                            , (inputs, tf.TensorShape([None, None, self.units]))])

                    inputs, states = self.decoder(inputs, states, training=training, timestamp_embedding=tar_timestamps,
                                                  iter_step=i)
                    tmp = self.fc(inputs)
                    out = tf.concat([out, tmp[:, -1:, :]], axis=1)
                return out

            if self.is_seq_continuous:
                inputs = enc_seq[:, -1:, :]
            else:
                inputs = tf.stack([tf.shape(enc_seq)[0], 1, self.units])
                inputs = tf.fill(inputs, 0.0)
            output = autoregress(inputs, states)
        elif self.gen_mode == "mlp":
            states = Flatten()(states[:, :, -1])
            output = self.fc(states)
        return output


class StationaryEncoder(Encoder):
    def __init__(self, layers, units, avg_window, seq_len=None, filters=None, kernel_size=3, rate=0.1):
        super().__init__(layers, units, seq_len=seq_len, filters=filters, kernel_size=kernel_size, rate=rate)
        self.decompose = SeriesDecompose(avg_window)

    def call(self, input_seq, training, timestamp_embedding=None):
        x = self.conv(input_seq)
        if timestamp_embedding is not None:
            x += timestamp_embedding
        if self.pos_vec is not None:
            x += self.pos_vec
        states = []
        for i in range(self.layers):
            x, state = self.gru_layers[i](x, training=training)
            x, _ = self.decompose(x)
            states.append(state)
        states = tf.stack(states, axis=-1)
        return x, states


class StationaryDecoder(Decoder):
    def __init__(self, layers, units, avg_window, seq_len=None, filters=None, kernel_size=3, rate=0.1):
        super().__init__(layers, units, seq_len=seq_len, filters=filters, kernel_size=kernel_size, rate=rate)
        self.decompose = SeriesDecompose(avg_window)

    def call(self, input_seq, initial_states, training, timestamp_embedding=None, iter_step=None):
        x = self.conv(input_seq)
        states = []
        if timestamp_embedding is not None:
            x += timestamp_embedding if iter_step is None else timestamp_embedding[:, iter_step:iter_step + 1, :]
        if self.pos_vec is not None:
            x += self.pos_vec if iter_step is None else self.pos_vec[:, iter_step:iter_step + 1, :]
        for i in range(self.layers):
            x, state = self.gru_layers[i](x, training=training, initial_state=initial_states[:, :, i])
            x, _ = self.decompose(x)
            states.append(state)
        states = tf.stack(states, axis=-1)
        return x, states


class StationaryConvGRU(ConvGRU):
    def __init__(self, num_layers, in_seq_len, in_dim, out_seq_len, out_dim, units, filters=None, kernel_size=3,
                 gen_mode='unistep',
                 is_seq_continuous=False, rate=0.1, avg_window=9):
        super().__init__(num_layers, in_seq_len, in_dim, out_seq_len, out_dim, units, filters=filters,
                         kernel_size=kernel_size,
                         gen_mode=gen_mode,
                         is_seq_continuous=is_seq_continuous, rate=rate)
        self.decompose = SeriesDecompose(avg_window)
        self.encoder = StationaryEncoder(num_layers, units, avg_window, seq_len=in_seq_len, filters=filters,
                                         kernel_size=kernel_size, rate=rate)
        self.decoder = StationaryDecoder(num_layers, units, avg_window, seq_len=out_seq_len, filters=filters,
                                         kernel_size=kernel_size, rate=rate)

    def call(self, input_seq, training, time_embedding_tuple=None):
        seasonality, _ = self.decompose(input_seq)
        out = super().call(seasonality, training, time_embedding_tuple=time_embedding_tuple)
        return out


if __name__ == '__main__':
    train_path_with_weather_info = os.path.sep.join(["../{}".format(parameter.csv_name)])
    data_with_weather_info = DataUtil(train_path=train_path_with_weather_info,
                                      val_path=None,
                                      test_path=None,
                                      normalise=0,
                                      label_col=parameter.target,
                                      feature_col=parameter.features,
                                      split_mode=parameter.split_mode,
                                      month_sep=parameter.test_month)
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
                         samples_per_day=dataUtil.samples_per_day,
                         using_timestamp_data=True)
    input_scalar = Input(shape=(src_len, len(parameter.features)))
    input_time = Input(shape=(src_len + shift + tar_len, len(time_embedding.vocab_size)))
    embedding = time_embedding.TimeEmbedding(output_dims=Config.embedding_filters, input_len=src_len, shift_len=shift,
                                             label_len=tar_len)(input_time)
    LR = StationaryConvGRU(num_layers=Config.layers, in_seq_len=src_len, in_dim=len(parameter.features),
                           out_seq_len=tar_len,
                           out_dim=len(parameter.target), units=Config.gru_units, filters=Config.embedding_filters,
                           gen_mode='unistep',
                           is_seq_continuous=True)(input_scalar, time_embedding_tuple=embedding)
    model = Model(inputs=[input_scalar, input_time], outputs=LR)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                  , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()])
    model.summary()
    tf.keras.backend.clear_session()
    history = model.fit(w2.trainData(addcloud=parameter.addAverage),
                        validation_data=w2.valData(addcloud=parameter.addAverage),
                        epochs=100, batch_size=5, callbacks=[parameter.earlystoper])
    for x, y in w2.trainData(addcloud=parameter.addAverage):
        c = model(x)
    model.summary()
    pass
