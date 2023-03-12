import tensorflow as tf


class Config():
    window_size = 17


class SeriesDecompose(tf.keras.layers.Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=self.window_size, strides=1, padding="same")

    def call(self, inputs, training):
        trend_component = self.avg_pool(inputs)
        seasonal_component = inputs - trend_component
        return seasonal_component, trend_component


class MovingZScoreNorm(tf.keras.layers.Layer):
    def __init__(self, window_size):
        super().__init__()
        assert window_size > 1
        self.window_size = window_size
        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=self.window_size, strides=1, padding="same")

    def call(self, inputs, training):
        # Compute the dimensions of inputs
        num_channels = int(inputs.shape[-1])
        seq_len = int(inputs.shape[1])

        # Compute the moving mean
        moving_mean = self.avg_pool(inputs)

        # Compute the moving std
        kernel = tf.ones([self.window_size, num_channels, num_channels])
        moving_variance = tf.nn.conv1d(tf.square(inputs - moving_mean), kernel, stride=1, padding='SAME') / (
                    self.window_size - 1)
        moving_std = tf.sqrt(moving_variance)

        # Compute z-score norm
        epsilon = 1e-7  # small constant that is set to address divide-by-0 problem
        output = (inputs - moving_mean) / (moving_std + epsilon)
        return output


if __name__ == '__main__':
    i = tf.keras.Input(shape=(20, 2))
    s, t = SeriesDecompose(window_size=5)(i)
    s = MovingZScoreNorm(window_size=5)(i)
    model = tf.keras.Model(inputs=[i], outputs=[s, t])
    model.summary()
    x = tf.random.normal([200, 20, 2])
    s, t = model(x)
    pass
