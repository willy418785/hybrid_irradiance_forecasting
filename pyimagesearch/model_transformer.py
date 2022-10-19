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
    def __init__(self, num_layers, d_model, num_heads, dff, seq_len, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len
        k_dim = int(d_model / num_heads)
        v_dim = k_dim
        self.embedding = Conv1D(filters=d_model, kernel_size=3, strides=1, padding="same", activation='elu')
        self.pos_encoding = positional_encoding(seq_len, d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, key_dim=k_dim, value_dim=v_dim, dff=dff, rate=rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, is_pooling, training, mask):
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :self.seq_len, :]

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
    def __init__(self, num_layers, d_model, num_heads, dff, seq_len, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len
        k_dim = int(d_model / num_heads)
        v_dim = k_dim
        self.embedding = Conv1D(filters=d_model, kernel_size=3, strides=1, padding="causal", activation='elu')
        self.pos_encoding = positional_encoding(seq_len, d_model)

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, key_dim=k_dim, value_dim=v_dim, dff=dff, rate=rate)
            for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, enc_output, is_pooling, training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, is_pooling, training)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, src_seq_len, tar_seq_len, src_dim, tar_dim, rate=0.1,
                 gen_mode="unistep", is_seq_continuous=False, is_pooling=False):
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
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               seq_len=src_seq_len, rate=rate)
        if gen_mode == 'unistep':
            self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                                   num_heads=num_heads, dff=dff,
                                   seq_len=src_seq_len + tar_seq_len, rate=rate)

        elif gen_mode == 'auto':
            self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                                   num_heads=num_heads, dff=dff,
                                   seq_len=tar_seq_len, rate=rate)
        self.final_layer = Dense(tar_dim)
        self.build(input_shape=(None, src_seq_len, src_dim))

    def call(self, inputs, training):
        enc_out = self.encoder(inputs, self.is_pooling, training, mask=None)
        if self.gen_mode == 'unistep':
            tar = tf.stack([tf.shape(inputs)[0], self.tar_seq_len, self.src_dim])
            tar = tf.fill(tar, 0.0)
            tar = tf.concat([inputs, tar], axis=1)
            dec_out = self.decoder(tar, enc_out, False, training)
            out = self.final_layer(dec_out)
            out = out[:, self.src_seq_len:, :]
        elif self.gen_mode == 'auto':
            @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, self.d_model), dtype=tf.float32)],
                         experimental_relax_shapes=True)
            def autoregress(tar):
                # print('[Side Effect] Retracing graph')
                for i in tf.range(self.tar_seq_len):
                    tf.autograph.experimental.set_loop_options(
                        shape_invariants=[(tar, tf.TensorShape([None, None, self.d_model]))])

                    dec_out = self.decoder(tar, enc_out, False, training)

                    tar = tf.concat([tar, dec_out[:, -1:, :]], axis=1)
                out = self.final_layer(tar[:, 1:, :])
                return out
            if self.is_seq_continuous:
                dec_input = enc_out[:, -1:, :]
            else:
                dec_input = tf.stack([tf.shape(inputs)[0], 1, self.d_model])
                dec_input = tf.fill(dec_input, 0.0)
            out = autoregress(dec_input)
        return out


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
    src_len = 540
    tar_len = 540
    dataUtil = data_with_weather_info
    w2 = WindowGenerator(input_width=src_len,
                         image_input_width=0,
                         label_width=tar_len,
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

                         batch_size=parameter.batchsize,
                         label_columns="ShortWaveDown")
    model = Transformer(num_layers=3, d_model=8, num_heads=4, dff=32, src_seq_len=src_len, tar_seq_len=tar_len, src_dim=len(parameter.features),
                        tar_dim=len(parameter.target), rate=0.1, gen_mode="unistep", is_seq_continuous=True, is_pooling=True)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam"
                  , metrics=[tf.metrics.MeanAbsoluteError()
            , tf.metrics.MeanAbsolutePercentageError()])

    model.summary()
    # tf.keras.backend.clear_session()
    # history = model.fit(w2.trainData(addcloud=parameter.addAverage),
    #                     validation_data=w2.valData(addcloud=parameter.addAverage),
    #                     epochs=100, batch_size=parameter.batchsize, callbacks=[parameter.earlystoper])
    for x, y in w2.trainData(addcloud=parameter.addAverage):
        c = model(x)
        pass
