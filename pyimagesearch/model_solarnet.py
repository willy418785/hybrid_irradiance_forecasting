import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model
def SolarNet(height, width, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
  inputShape = (height, width, depth)
  chanDim = -1
  
  # define the model input
  inputs = Input(shape=inputShape)

  x = Conv2D(64, (3, 3), padding="same")(inputs)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Conv2D(64, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)

  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  x = Conv2D(128, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Conv2D(128, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)

  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  x = Conv2D(256, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Conv2D(256, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Conv2D(256, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)

  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  x = Conv2D(512, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Conv2D(512, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Conv2D(512, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)

  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  x = Conv2D(512, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Conv2D(512, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Conv2D(512, (3, 3), padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
  x = Flatten()(x)
  x = Dense(256)(x)

  x = Activation("relu")(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Dropout(0.5)(x)

  x = Dense(5)(x)
  x = Activation("relu")(x)
  if regress:
    x = Dense(1)(x)
  # construct the CNN
  model = Model(inputs, x)
  
  # return the CNN
  return model

'''
class SolarNet(keras.Model):
  def __init__(self, width, height, depth, nums_class=1):
    super(SolarNet,self).__init__()
    self.inputShape = (height, width, depth)
    self.inputs = layers.Input(shape=self.inputShape)
    self.model = Sequential()
    self.conv_1_2 = layers.Conv2D(64,(3,3),strides=2,padding='same')
    self.conv_3_4 = layers.Conv2D(128,(3,3),strides=2,padding='same')
    self.conv_5_7 = layers.Conv2D(256,(3,3),strides=2,padding='same')
    self.conv_8_13 = layers.Conv2D(512,(3,3),strides=2,padding='same')
    self.max_pool = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
    self.bn = layers.BatchNormalization()
    self.act_relu = layers.Activation('relu')
     
    # self.avg_pool = layers.AveragePooling2D(pool_size=(7, 7))
    self.fc_model_1 = layers.Dense(256)
    self.fc_model_1 = layers.Dense(1)


  def call(self, inputs, training=None):
    x = self.model(self.inputs)
    x = self.conv_1_2(x)
    x = self.bn(x)
    x = self.act_relu(x)
    x = self.conv_1_2(x)
    x = self.bn(x)
    x = self.act_relu(x)

    x = self.max_pool(x)
    x = self.conv_3_4(x)
    x = self.bn(x)
    x = self.act_relu(x)
    x = self.conv_3_4(x)
    x = self.bn(x)
    x = self.act_relu(x)

    x = self.max_pool(x)
    x = self.conv_5_7(x)
    x = self.bn(x)
    x = self.act_relu(x)
    x = self.conv_5_7(x)
    x = self.bn(x)
    x = self.act_relu(x)
    x = self.conv_5_7(x)
    x = self.bn(x)
    x = self.act_relu(x)

    x = self.max_pool(x)
    x = self.conv_8_13(x)
    x = self.bn(x)
    x = self.act_relu(x)
    x = self.conv_8_13(x)
    x = self.bn(x)
    x = self.act_relu(x)
    x = self.conv_8_13(x)
    x = self.bn(x)
    x = self.act_relu(x)

    x = self.max_pool(x)
    x = self.conv_8_13(x)
    x = self.bn(x)
    x = self.act_relu(x)
    x = self.conv_8_13(x)
    x = self.bn(x)
    x = self.act_relu(x)
    x = self.conv_8_13(x)
    x = self.bn(x)
    x = self.act_relu(x)

    x = self.max_pool(x)
    x = self.fc_model_1(x)
    x = self.fc_model_2(x)
    return x'''