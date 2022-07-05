from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling3D, Activation, Dropout, Dense, LSTM, Reshape, ConvLSTM2D, Input, Conv3D
from tensorflow.keras.layers import AveragePooling3D, BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
def convlstm(SequenceLength, height, width, depth, regress=False):
  # use simple CNN structure
  in_shape = (SequenceLength, height, width, 3)
  inputs = Input(shape=in_shape)
  
  x = ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True)(inputs)
  x = Activation("relu")(x)
  x = MaxPooling3D(pool_size=(1, 2, 2))(x)
  
  x = ConvLSTM2D(64, kernel_size=(5, 5), padding='valid', return_sequences=True)(x)
  x = MaxPooling3D(pool_size=(1, 2, 2))(x)

  x = ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
  x = Activation("relu")(x)
  x = ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
  x = Activation("relu")(x)

  x = ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True)(x)
  x = MaxPooling3D(pool_size=(1, 2, 2))(x)
  x = Dense(320)(x)
  x = Activation("relu")(x)
  x = Dropout(0.5)(x)

  # print(x.shape)    #(none,1,3,3,320)
  out_shape = x.shape
  # print('====Model shape: ', out_shape)
  x = Reshape((SequenceLength, out_shape[2] * out_shape[3] * out_shape[4]))(x)
  x = LSTM(64, return_sequences=False)(x)
  x = Dropout(0.5)(x)
  x = Dense(5, activation='softmax')(x)

  if regress:
    x = Dense(1)(x)
  # construct the CNN
  model = Model(inputs, x)
  # model structure summary
  # print(model.summary())
  return model
