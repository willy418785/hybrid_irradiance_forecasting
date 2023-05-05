import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


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
    def __init__(self, window_size,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        assert window_size > 1
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.window_size = window_size
        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=self.window_size, strides=1, padding="same")

    def build(self, input_shape):
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=(input_shape[-1]),
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=(input_shape[-1]),
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.beta = None
        super().build(input_shape)

    def call(self, inputs, training):
        # Compute the dimensions of inputs
        num_channels = int(inputs.shape[-1])

        # Compute the moving mean
        moving_mean = self.avg_pool(inputs)

        # Compute the moving std
        kernel = tf.ones([self.window_size, num_channels, num_channels])
        moving_variance = tf.nn.conv1d(tf.square(inputs - moving_mean), kernel, stride=1, padding='SAME') / (
                self.window_size - 1)
        moving_std = tf.sqrt(moving_variance)

        # Compute z-score norm
        output = (inputs - moving_mean) / (moving_std + self.epsilon)
        if self.gamma is not None:
            output = output * self.gamma
        if self.beta is not None:
            output = output + self.beta
        return output, moving_mean


if __name__ == '__main__':
    i = tf.keras.Input(shape=(20, 2))
    s, t = SeriesDecompose(window_size=5)(i)
    s = MovingZScoreNorm(window_size=5)(i)
    model = tf.keras.Model(inputs=[i], outputs=[s, t])
    model.summary()
    x = tf.random.normal([200, 20, 2])
    s, t = model(x)
    pass
