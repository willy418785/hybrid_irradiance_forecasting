import tensorflow as tf

class Config():
    filters = 32
    kernel_size = 3

class SplitInputByDay(tf.keras.layers.Layer):
    def __init__(self, n_days, n_samples):
        super().__init__()
        self.n_days = n_days
        self.n_samples = n_samples
        self.reshape = tf.keras.layers.Reshape((n_days, n_samples, -1))

    def call(self, inputs, training):
        channels = tf.shape(inputs)[-1]
        x = self.reshape(inputs, training=training)
        # x = tf.ensure_shape(x, [None, self.n_days, self.n_samples, channels])
        return x

class MultipleDaysConvEmbed(tf.keras.layers.Layer):
    def __init__(self, filters, filter_size, n_days, n_samples):
        super().__init__()
        self.filters = filters
        self.filter_size = filter_size
        self.pad_size = int(filter_size/2)
        self.n_days = n_days
        self.n_samples = n_samples
        self.pad = tf.keras.layers.ZeroPadding2D(padding=(0, self.pad_size))
        self.embed = tf.keras.layers.Conv2D(filters=filters, kernel_size=(n_days, filter_size), strides=1, padding='valid',activation='elu')
        self.reshape = tf.keras.layers.Reshape((n_samples, -1))

    def call(self, inputs, training):
        channels = tf.shape(inputs)[-1]
        x = self.pad(inputs, training=training)
        x = self.embed(x, training=training)
        x = self.reshape(x, training=training)
        # x = tf.ensure_shape(x, [None, self.n_samples, channels])
        return x

if __name__ == '__main__':
    model = tf.keras.Sequential([SplitInputByDay(2, 10), MultipleDaysConvEmbed(10, 5, 2, 10)])
    model.build(input_shape=(None, 20, 2))
    model.summary()
    x = tf.random.normal([200, 20, 2])
    y = model(x)
    pass
