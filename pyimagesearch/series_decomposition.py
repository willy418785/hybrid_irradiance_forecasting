import tensorflow as tf


class Config():
    window_size = 8


class SeriesDecompose(tf.keras.layers.Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=self.window_size, strides=1, padding="same")

    def call(self, inputs, training):
        trend_component = self.avg_pool(inputs)
        seasonal_component = inputs - trend_component
        return seasonal_component, trend_component

if __name__ == '__main__':
    i = tf.keras.Input(shape=(20, 2))
    s, t = SeriesDecompose(window_size=5)(i)
    model = tf.keras.Model(inputs=[i], outputs=[s,t])
    model.summary()
    x = tf.random.normal([200, 20, 2])
    s, t = model(x)
    pass
