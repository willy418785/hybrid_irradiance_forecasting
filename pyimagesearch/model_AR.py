import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from pyimagesearch import parameter
from pyimagesearch.datautil import DataUtil

gen_modes = ['unistep', 'auto', "mlp"]


class ARModel(tf.keras.Model):
    def __init__(self, ar, tar_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ar = ar
        self.tar_len = tar_len

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        len = tf.shape(y)[1]
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, tar_len=len)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        len = tf.shape(y)[1]
        # Compute predictions
        y_pred = self(x, training=False, tar_len=len)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training, tar_len=None):
        if tar_len is None:
            tar_len = self.tar_len
        dims = int(inputs.shape.as_list()[-1])
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, dims), dtype=tf.float32)],
                     experimental_relax_shapes=True)
        def autoregress(tar):
            out = tf.stack([tf.shape(inputs)[0], 0, dims])
            out = tf.fill(out, 0.0)
            for i in tf.range(tar_len):
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[(out, tf.TensorShape([None, None, dims]))])
                y = self.ar(tar, training=training)
                tar = tf.concat([tar, y], axis=1)
                out = tf.concat([out, y], axis=1)
            return out

        output = autoregress(tf.cast(inputs, tf.float32))
        # output = self.ar(inputs, training=training, tar_len=tar_len)
        return output


class AR(tf.keras.layers.Layer):
    def __init__(self, order):
        super().__init__()
        self.order = order
        self.linear = Dense(1)

    def call(self, inputs, training):
        temporal_last = tf.transpose(inputs, perm=[0, 2, 1])
        out = self.linear(temporal_last[:, :, -self.order:])
        final = tf.transpose(out, perm=[0, 2, 1])
        return final


class ChannelIndependentAR(tf.keras.layers.Layer):
    def __init__(self, order, src_dims):
        super().__init__()
        self.order = order
        self.src_dims = src_dims
        self.linears = [Dense(1) for _ in range(src_dims)]

    def call(self, inputs, training):
        temporal_last = tf.transpose(inputs, perm=[0, 2, 1])
        out = [self.linears[i](temporal_last[:, i, -self.order:]) for i in range(self.src_dims)]
        stacked = tf.stack(out, axis=1)
        final = tf.transpose(stacked, perm=[0, 2, 1])
        return final


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
    src_len = parameter.data_params.input_width
    tar_len = parameter.data_params.label_width
    shift = parameter.data_params.shifted_width
    order = parameter.model_params.bypass_params.order

    from pyimagesearch.windowsGenerator import WindowGenerator

    teacher_forcing_w = WindowGenerator(input_width=src_len,
                                        image_input_width=0,
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
    test_w = WindowGenerator(input_width=src_len,
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
    input = Input(shape=(src_len, len(parameter.data_params.features)))
    ar = ChannelIndependentAR(order=order, src_dims=len(parameter.data_params.features))
    output = ar(input)
    model = ARModel(ar=ar, tar_len=tar_len, inputs=input, outputs=output)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer="Adam",
                  metrics=[tf.metrics.MeanAbsoluteError(),
                           tf.metrics.MeanAbsolutePercentageError()])
    model.summary()
    tf.keras.backend.clear_session()
    history = model.fit(teacher_forcing_w.train(1,
                                 addcloud=parameter.data_params.addAverage,
                                 using_timestamp_data=False,
                                 is_shuffle=parameter.data_params.is_using_shuffle),
                        validation_data=teacher_forcing_w.val(1,
                                               addcloud=parameter.data_params.addAverage,
                                               using_timestamp_data=False,
                                               is_shuffle=parameter.data_params.is_using_shuffle),
                        epochs=1)
    for x, y in test_w.test(test_w.samples_per_day, addcloud=False, using_timestamp_data=False):
        c = model(x[:1, :, :])
        print(c.shape)
    model.summary()
    pass
