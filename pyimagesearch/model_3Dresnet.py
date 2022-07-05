import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model
class ResBlock(layers.Layer):
  def __init__(self, filter_nums, strides=1, residual_path=False):
      super(ResBlock, self).__init__()

      self.conv_1 = layers.Conv3D(filter_nums,(1,3,3),strides=strides,padding='same')
      self.bn_1 = layers.BatchNormalization()
      self.act_relu = layers.Activation('relu')

      self.conv_2 = layers.Conv3D(filter_nums,(1,3,3),strides=1,padding='same')
      self.bn_2 = layers.BatchNormalization()
      
      if strides !=1:
        self.block = Sequential()
        self.block.add(layers.Conv3D(filter_nums,(1,1,1),strides=strides))
      else:
        self.block = lambda x:x


  def call(self, inputs, training=None):

      x = self.conv_1(inputs)
      x = self.bn_1(x, training=training)
      x = self.act_relu(x)
      x = self.conv_2(x)
      x = self.bn_2(x,training=training)
      
      identity = self.block(inputs)
      outputs = layers.add([x,identity])
      outputs = tf.nn.relu(outputs)

      return outputs

class ResNet(keras.Model):
  def __init__(self, layers_dims, nums_class=1, regress=False):
    super(ResNet,self).__init__()
    self.model = Sequential([layers.Conv3D(64,(1,3,3),strides=(1,1,1)),
                             layers.BatchNormalization(),
                             layers.Activation('relu'),
                             layers.MaxPooling3D(pool_size=(1,2,2),strides=(1,1,1),padding='same')])

    self.layer_1 = self.ResNet_build(64,layers_dims[0])
    self.layer_2 = self.ResNet_build(128,layers_dims[1],strides=2)   
    self.layer_3 = self.ResNet_build(256,layers_dims[2],strides=2) 
    self.layer_4 = self.ResNet_build(512,layers_dims[3],strides=2)   

    self.avg_pool = layers.GlobalAveragePooling3D()
    self.fc_model_1 = layers.Dense(nums_class)
    self.fc_model_2 = layers.Dense(1)
    self.regress = regress
    
  def call(self, inputs, training=None):
    x = self.model(inputs)
    x = self.layer_1(x)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)
    x = self.avg_pool(x)
    x = self.fc_model_1(x)

    # if self.regress:
    #   x = self.fc_model_2(x)
    return x

  def ResNet_build(self,filter_nums,block_nums,strides=1):
    build_model = Sequential()
    build_model.add(ResBlock(filter_nums,strides))
    for _ in range(1,block_nums):
      build_model.add(ResBlock(filter_nums,strides=1))
    return build_model