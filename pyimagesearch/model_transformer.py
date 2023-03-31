import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Concatenate, Dropout, Dense, Flatten, \
    Input, Add
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, LayerNormalization, Reshape, ZeroPadding2D, \
    MultiHeadAttention
from tensorflow.keras.models import Model

from pyimagesearch import parameter, time_embedding
from pyimagesearch.datautil import DataUtil
from pyimagesearch.series_decomposition import SeriesDecompose, MovingZScoreNorm
from pyimagesearch.windowsGenerator import WindowGenerator

gen_modes = ['unistep', 'auto', "mlp"]


class Config():
    layers = 1
    d_model = 32
    n_heads = 1
    dff = d_model * 4
    embedding_kernel_size = 3
    dropout_rate = 0.1


def feed_forward(d_model, dff, dropout):
    return tf.keras.Sequential([
        Dense(dff, activation='elu'),  # (batch_size, seq_len, dff)
        Dense(d_model),  # (batch_size, seq_len, d_model)
        Dropout(dropout)
    ])


def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def attention_masking(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, key_dim, value_dim, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=rate)
        self.ff = feed_forward(d_model, dff, rate)
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.pooling = MaxPooling1D(pool_size=2, strides=2, padding='same')

    def call(self, x, is_pooling, training, mask):
        attn_output = self.mha(x, x, x, mask, training=training)  # (batch_size, input_seq_len, d_model)
        out1 = self.ln1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ff_output = self.ff(out1)  # (batch_size, input_seq_len, d_model)
        out2 = self.ln2(out1 + ff_output)  # (batch_size, input_seq_len, d_model)
        if is_pooling:
            out2 = self.pooling(out2)
        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, seq_len, kernel_size=3, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.k_dim = int(d_model / num_heads)
        self.v_dim = self.k_dim
        self.embedding = Conv1D(filters=d_model, kernel_size=kernel_size, strides=1, padding="same",
                                activation='elu')
        self.pos_encoding = positional_encoding(seq_len, d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, key_dim=self.k_dim, value_dim=self.v_dim, dff=dff,
                         rate=rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, is_pooling, training, mask, timestamp_embedding=None):
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :self.seq_len, :]
        if timestamp_embedding is not None:
            x += timestamp_embedding
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, is_pooling, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, key_dim, value_dim, dff, rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=rate)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=rate)
        self.ff = feed_forward(d_model, dff, rate)
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.ln3 = LayerNormalization()
        self.pooling = MaxPooling1D(pool_size=2, strides=2, padding='same')

    def call(self, x, enc_output, is_pooling, training):
        look_ahead_mask = attention_masking(tf.shape(x)[1])
        attn1 = self.mha1(x, x, x, look_ahead_mask, training=training)  # (batch_size, target_seq_len, d_model)
        out1 = self.ln1(attn1 + x)

        if is_pooling:
            out1 = self.pooling(out1)

        attn2 = self.mha2(out1, enc_output, training=training)  # (batch_size, target_seq_len, d_model)
        out2 = self.ln2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ff_output = self.ff(out2)  # (batch_size, target_seq_len, d_model)
        out3 = self.ln3(ff_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, seq_len, kernel_size=3, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.k_dim = int(d_model / num_heads)
        self.v_dim = self.k_dim
        self.embedding = Conv1D(filters=d_model, kernel_size=kernel_size, strides=1, padding="causal",
                                activation='elu')
        self.pos_encoding = positional_encoding(seq_len, d_model)

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, key_dim=self.k_dim, value_dim=self.v_dim, dff=dff,
                         rate=rate)
            for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, enc_output, is_pooling, training, timestamp_embedding=None, iter_step=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        if timestamp_embedding is not None:
            x += timestamp_embedding if iter_step is None else timestamp_embedding[:, :iter_step + 1, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, is_pooling, training)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, src_seq_len, tar_seq_len, src_dim, tar_dim, kernel_size=3,
                 rate=0.1,
                 gen_mode="unistep", is_seq_continuous=False, is_pooling=False, token_len=None):
        assert gen_mode in gen_modes
        super().__init__()
        self.d_model = d_model
        self.gen_mode = gen_mode
        self.src_seq_len = src_seq_len
        self.tar_seq_len = tar_seq_len
        self.src_dim = src_dim
        self.tar_dim = tar_dim
        self.is_seq_continuous = is_seq_continuous
        self.is_pooling = is_pooling
        self.kernel_size = kernel_size
        self.rate = rate
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               seq_len=src_seq_len, kernel_size=kernel_size, rate=rate)
        if self.gen_mode == 'unistep':
            if token_len is not None and is_seq_continuous:
                assert type(token_len) is int
                self.token_len = token_len
            else:
                self.token_len = 0
            self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                                   num_heads=num_heads, dff=dff,
                                   seq_len=token_len + tar_seq_len, kernel_size=kernel_size, rate=rate)
        elif self.gen_mode == 'auto':
            self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                                   num_heads=num_heads, dff=dff,
                                   seq_len=tar_seq_len, kernel_size=kernel_size, rate=rate)
        elif self.gen_mode == 'mlp':
            self.decoder = Sequential([Dense(tar_seq_len * tar_dim), Reshape((tar_seq_len, tar_dim))])
        self.final_layer = Dense(tar_dim)
        # self.build(input_shape=(None, src_seq_len, src_dim))

    def call(self, inputs, training, time_embedding_tuple=None):
        src_timestamps, shift_timestamps, tar_timestamps = time_embedding_tuple if time_embedding_tuple is not None else (
            None, None, None)
        enc_out = self.encoder(inputs, self.is_pooling, training, mask=None, timestamp_embedding=src_timestamps)
        if self.gen_mode == 'unistep':
            tar = tf.stack([tf.shape(inputs)[0], self.tar_seq_len, self.src_dim])
            tar = tf.fill(tar, 0.0)
            decoder_timestamps = tar_timestamps
            if self.token_len > 0 and self.is_seq_continuous:
                tar = tf.concat([inputs[:, -self.token_len:, :], tar], axis=1)
                if time_embedding_tuple is not None:
                    decoder_timestamps = tf.concat([src_timestamps[:, -self.token_len:, :], tar_timestamps], axis=1)
            dec_out = self.decoder(tar, enc_out, False, training, timestamp_embedding=decoder_timestamps)
            out = self.final_layer(dec_out)
            out = out[:, -self.tar_seq_len:, :]
        elif self.gen_mode == 'auto':
            @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, self.d_model), dtype=tf.float32)],
                         experimental_relax_shapes=True)
            def autoregress(tar):
                # print('[Side Effect] Retracing graph')
                for i in tf.range(self.tar_seq_len):
                    tf.autograph.experimental.set_loop_options(
                        shape_invariants=[(tar, tf.TensorShape([None, None, self.d_model]))])

                    dec_out = self.decoder(tar, enc_out, is_pooling=False, training=training,
                                           timestamp_embedding=tar_timestamps, iter_step=i)

                    tar = tf.concat([tar, dec_out[:, -1:, :]], axis=1)
                out = self.final_layer(tar[:, 1:, :])
                return out

            if self.is_seq_continuous:
                dec_input = enc_out[:, -1:, :]
            else:
                dec_input = tf.stack([tf.shape(inputs)[0], 1, self.d_model])
                dec_input = tf.fill(dec_input, 0.0)
            out = autoregress(dec_input)
        elif self.gen_mode == "mlp":
            flattened = Flatten()(enc_out)
            dec_out = self.decoder(flattened)
            out = self.final_layer(dec_out)

        return out


class StationaryEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, key_dim, value_dim, dff, rate=0.1, avg_window=9):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=rate)
        self.ff = feed_forward(d_model, dff, rate)
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.pooling = MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.decompose = SeriesDecompose(avg_window)

    def call(self, x, is_pooling, training, mask):
        attn_output = self.mha(x, x, x, mask, training=training)  # (batch_size, input_seq_len, d_model)
        out1 = self.ln1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        season1, _ = self.decompose(out1)
        ff_output = self.ff(season1)  # (batch_size, input_seq_len, d_model)
        out2 = self.ln2(season1 + ff_output)  # (batch_size, input_seq_len, d_model)
        season2, _ = self.decompose(out2)
        if is_pooling:
            season2 = self.pooling(season2)
        return season2


class StationaryDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, key_dim, value_dim, dff, rate=0.1, avg_window=9):
        super().__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=rate)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=rate)
        self.ff = feed_forward(d_model, dff, rate)
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.ln3 = LayerNormalization()
        self.pooling = MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.decompose = SeriesDecompose(avg_window)

    def call(self, x, enc_output, is_pooling, training):
        look_ahead_mask = attention_masking(tf.shape(x)[1])
        attn1 = self.mha1(x, x, x, look_ahead_mask, training=training)  # (batch_size, target_seq_len, d_model)
        out1 = self.ln1(attn1 + x)
        season1, _ = self.decompose(out1)
        if is_pooling:
            season1 = self.pooling(season1)

        attn2 = self.mha2(season1, enc_output, training=training)  # (batch_size, target_seq_len, d_model)
        out2 = self.ln2(attn2 + season1)  # (batch_size, target_seq_len, d_model)
        season2, _ = self.decompose(out2)

        ff_output = self.ff(season2)  # (batch_size, target_seq_len, d_model)
        out3 = self.ln3(ff_output + season2)  # (batch_size, target_seq_len, d_model)
        season3, _ = self.decompose(out3)
        return season3


class StationaryTransformer(Transformer):
    def __init__(self, num_layers, d_model, num_heads, dff, src_seq_len, tar_seq_len, src_dim, tar_dim, kernel_size=3,
                 rate=0.1,
                 gen_mode="unistep", is_seq_continuous=False, is_pooling=False, token_len=None, avg_window=9):
        assert gen_mode in gen_modes
        super().__init__(num_layers, d_model, num_heads, dff, src_seq_len, tar_seq_len, src_dim, tar_dim,
                         kernel_size=kernel_size, rate=rate,
                         gen_mode=gen_mode, is_seq_continuous=is_seq_continuous, is_pooling=is_pooling,
                         token_len=token_len)
        self.avg_window = avg_window
        self.decompose = SeriesDecompose(avg_window)
        self.encoder.enc_layers = [
            StationaryEncoderLayer(d_model=d_model, num_heads=num_heads, key_dim=self.encoder.k_dim,
                                   value_dim=self.encoder.v_dim,
                                   dff=dff, rate=rate, avg_window=avg_window) for _ in range(num_layers)]
        if self.gen_mode == 'unistep' or self.gen_mode == 'auto':
            self.decoder.dec_layers = [
                StationaryDecoderLayer(d_model=d_model, num_heads=num_heads, key_dim=self.decoder.k_dim,
                                       value_dim=self.decoder.v_dim,
                                       dff=dff, rate=rate, avg_window=avg_window) for _ in range(num_layers)]
    def call(self, inputs, training, time_embedding_tuple=None):
        seasonality, _ = self.decompose(inputs)
        out = super().call(seasonality, training, time_embedding_tuple=time_embedding_tuple)
        return out


class MovingZScoreNormEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, key_dim, value_dim, dff, rate=0.1, avg_window=9):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=rate)
        self.ff = feed_forward(d_model, dff, rate)
        self.zn1 = MovingZScoreNorm(avg_window)
        self.zn2 = MovingZScoreNorm(avg_window)
        self.pooling = MaxPooling1D(pool_size=2, strides=2, padding='same')

    def call(self, x, is_pooling, training, mask):
        attn_output = self.mha(x, x, x, mask, training=training)  # (batch_size, input_seq_len, d_model)
        out1, _ = self.zn1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ff_output = self.ff(out1)  # (batch_size, input_seq_len, d_model)
        out2, _ = self.zn2(ff_output + out1)  # (batch_size, input_seq_len, d_model)
        if is_pooling:
            out2 = self.pooling(out2)
        return out2


class MovingZScoreNormDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, key_dim, value_dim, dff, rate=0.1, avg_window=9):
        super().__init__()
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=rate)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=rate)
        self.ff = feed_forward(d_model, dff, rate)
        self.zn1 = MovingZScoreNorm(avg_window)
        self.zn2 = MovingZScoreNorm(avg_window)
        self.zn3 = MovingZScoreNorm(avg_window)
        self.pooling = MaxPooling1D(pool_size=2, strides=2, padding='same')

    def call(self, x, enc_output, is_pooling, training):
        look_ahead_mask = attention_masking(tf.shape(x)[1])
        attn1 = self.mha1(x, x, x, look_ahead_mask, training=training)  # (batch_size, target_seq_len, d_model)
        out1, _ = self.zn1(attn1 + x)
        if is_pooling:
            out1 = self.pooling(out1)

        attn2 = self.mha2(out1, enc_output, training=training)  # (batch_size, target_seq_len, d_model)
        out2, _ = self.zn2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ff_output = self.ff(out2)  # (batch_size, target_seq_len, d_model)
        out3, _ = self.zn3(ff_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3


class MovingZScoreNormTransformer(Transformer):
    def __init__(self, num_layers, d_model, num_heads, dff, src_seq_len, tar_seq_len, src_dim, tar_dim, kernel_size=3,
                 rate=0.1,
                 gen_mode="unistep", is_seq_continuous=False, is_pooling=False, token_len=None, avg_window=9):
        assert gen_mode in gen_modes
        super().__init__(num_layers, d_model, num_heads, dff, src_seq_len, tar_seq_len, src_dim, tar_dim,
                         kernel_size=kernel_size, rate=rate,
                         gen_mode=gen_mode, is_seq_continuous=is_seq_continuous, is_pooling=is_pooling,
                         token_len=token_len)
        self.avg_window = avg_window
        self.decompose = MovingZScoreNorm(avg_window)
        self.encoder.enc_layers = [
            MovingZScoreNormEncoderLayer(d_model=d_model, num_heads=num_heads, key_dim=self.encoder.k_dim,
                                         value_dim=self.encoder.v_dim,
                                         dff=dff, rate=rate, avg_window=avg_window) for _ in range(num_layers)]
        if self.gen_mode == 'unistep' or self.gen_mode == 'auto':
            self.decoder.dec_layers = [
                MovingZScoreNormDecoderLayer(d_model=d_model, num_heads=num_heads, key_dim=self.decoder.k_dim,
                                             value_dim=self.decoder.v_dim,
                                             dff=dff, rate=rate, avg_window=avg_window) for _ in range(num_layers)]

    def call(self, inputs, training, time_embedding_tuple=None):
        seasonality, trend = self.decompose(inputs)
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
                         samples_per_day=dataUtil.samples_per_day)
    input_scalar = Input(shape=(src_len, len(parameter.features)))
    input_time = Input(shape=(src_len + shift + tar_len, len(time_embedding.vocab_size)))
    embedding = time_embedding.TimeEmbedding(output_dims=Config.d_model, input_len=src_len, shift_len=shift,
                                             label_len=tar_len)(input_time)
    LR = MovingZScoreNormTransformer(num_layers=Config.layers, d_model=Config.d_model, num_heads=Config.n_heads,
                               dff=Config.dff,
                               src_seq_len=src_len, tar_seq_len=tar_len,
                               src_dim=len(parameter.features),
                               tar_dim=len(parameter.target), rate=0.1, gen_mode="unistep", is_seq_continuous=True,
                               is_pooling=True, token_len=5)(input_scalar, time_embedding_tuple=embedding)

    model = Model(inputs=[input_scalar, input_time], outputs=LR)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                  , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()])
    model.summary()
    tf.keras.backend.clear_session()
    history = model.fit(w2.train(w2.samples_per_day, addcloud=False, using_timestamp_data=True, is_shuffle=False),
                        validation_data=w2.val(w2.samples_per_day, addcloud=False, using_timestamp_data=True, is_shuffle=False),
                        epochs=100, batch_size=5, callbacks=[parameter.earlystoper])
    for x, y in w2.train(w2.samples_per_day, addcloud=False, using_timestamp_data=True, is_shuffle=False):
        c = model(x)
    model.summary()
    pass
