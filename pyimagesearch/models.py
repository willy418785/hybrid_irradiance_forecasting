# import the necessary packages
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Concatenate, Dropout, Dense, Flatten, Input, Add
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, LayerNormalization, Reshape, ZeroPadding2D, MultiHeadAttention
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from pyimagesearch import parameter

def cnnlstm(SequenceLength, regress=False):
	# define our MLP network
	inputs = Input(shape=(SequenceLength, 1), name='input')
	
	# 64 filters, 10 kernel size
	x = Conv1D(8, 3, activation='relu')(inputs)
	x = MaxPooling1D()(x)
	x = BatchNormalization()(x)
	'''
	x = Conv1D(16, 3, activation='relu')(x)
	x = MaxPooling1D()(x)
	x = BatchNormalization()(x)
	
	x = Conv1D(32, 3, activation='relu')(x)
	x = MaxPooling1D()(x)
	x = BatchNormalization()(x)'''

	lstm_2 = GRU(5, return_sequences=True, name="gru1")(x)
	out = GRU(1,  return_sequences=False, name="gru2")(lstm_2)

	# model.add(Flatten())
	# model.add(Dense(64, activation='relu', name='dense_1'))
	# model.add(BatchNormalization())
	# model.add(Dropout(dropout))
	# model.add(Dense(1, activation="linear"))
	# return our model
	model = Model(inputs, out)

	print(model.summary())
	return model

def create_mlp(SequenceLength, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=SequenceLength, activation="relu"))
	model.add(Dense(4, activation="relu"))

	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))

	# return our model
	
	return model


###################################### use #########################

def cnnlstm_david(SequenceLength, num, predict_length=parameter.label_width, regress=False, expand=False):
	inputs = Input(shape=(SequenceLength, num))			#len(parameter.inputs)
	# x = Conv1D(filters=100, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
	# x = ZeroPadding1D(padding=2)(x)
	# whole_seq_output, final_memory_state = GRU(5, kernel_initializer="glorot_uniform")(x)
	# result = Dense(5)(final_memory_state)
	

	x = tf.expand_dims(inputs,axis=-1)
	print(x)
	x = Conv2D(filters=100, kernel_size=[5, 1], strides=1,padding="valid", activation='relu')(x)
	x = ZeroPadding2D(padding=(2, 0))(x)	
	print(x)
	x = Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)
	print(x)
	# x = K.permute_dimensions(x, (0, 1, 3, 2))   #(None,120,5,1)
	# x = tf.squeeze(x, axis=-1)	
	whole_seq_output, final_memory_state = GRU(5, kernel_initializer="glorot_uniform",return_state=True,return_sequences=True,activation='relu')(x)							
	result = Dense(predict_length)(final_memory_state)
	
	if regress:
		result = Dense(1)(result)
		
	if expand:
		result = tf.expand_dims(result, axis=-1)

	result = tf.expand_dims(result, axis=-1)


	model = Model(inputs, result)
	
	# print(model.summary())
	return model

def simple_transformer(model_dim, heads, model_depth, input_dim, target_dim, cloud_data_dim = None, seq_len = None, out_seq_len = None):
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

	k_dim = int(model_dim / heads)
	v_dim = k_dim
	input_data = Input(shape=(seq_len, input_dim))
	if cloud_data_dim != None:
		cloud_data_input = Input(shape=(seq_len, cloud_data_dim))
		inputs = [input_data, cloud_data_input]
		concat = Concatenate()([input_data, cloud_data_input])
	else:
		inputs = input_data
		concat = input_data
	x = Conv1D(filters=model_dim, kernel_size=3, strides=1, padding="causal", activation='elu')(concat)
	input_pos_enocoded = positional_encoding(seq_len, model_dim)
	input_pos_enocoded = tf.repeat(input_pos_enocoded, tf.shape(x)[0], axis=0)
	x = Add()([x, input_pos_enocoded])
	for _ in range(model_depth):
		# encoder
		original = x
		x = MultiHeadAttention(num_heads=heads, key_dim=k_dim, value_dim=v_dim)(x, x)
		x = Add()([original, x])
		x = LayerNormalization()(x)
		original = x
		x = Dense(2 * model_dim, "elu")(x)
		x = Dense(model_dim)(x)
		x = Add()([original, x])
		x = LayerNormalization()(x)
	enc_out = x
	# generate decoder's input (zero tensor)
	decoder_input = tf.stack([tf.shape(x)[0], out_seq_len, input_dim + (cloud_data_dim if cloud_data_dim is not None else 0)])
	decoder_input = tf.fill(decoder_input, 0.0)
	decoder_input = tf.concat([concat, decoder_input], axis=1)
	decoder_input = Conv1D(filters=model_dim, kernel_size=3, strides=1, padding="causal", activation='elu')(decoder_input)
	decoder_input_pos_enocoded = positional_encoding(seq_len+out_seq_len, model_dim)
	decoder_input_pos_enocoded = tf.repeat(decoder_input_pos_enocoded, tf.shape(x)[0], axis=0)
	x = Add()([decoder_input, decoder_input_pos_enocoded])
	for _ in range(model_depth):
		# decoder
		original = x
		mask = tf.repeat(attention_masking(tf.shape(x)[1])[tf.newaxis, :], tf.shape(x)[0], axis=0)
		x = MultiHeadAttention(num_heads=heads, key_dim=k_dim, value_dim=v_dim)(x, x, x, mask)
		x = Add()([original, x])
		x = LayerNormalization()(x)
		original = x
		x = MultiHeadAttention(num_heads=heads, key_dim=k_dim, value_dim=v_dim)(x, enc_out)
		x = Add()([original, x])
		x = LayerNormalization()(x)
		original = x
		x = Dense(2 * model_dim, "elu")(x)
		x = Dense(model_dim)(x)
		x = Add()([original, x])
		x = LayerNormalization()(x)
	out = Dense(target_dim)(x)
	out = out[:, seq_len:, :]
	model = Model(inputs, out)
	return model
'''Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           [(None, 5, 1)]            0
_________________________________________________________________
tf.expand_dims (TFOpLambda)  (None, 5, 1, 1)           0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1, 1, 100)         600
_________________________________________________________________
zero_padding2d (ZeroPadding2 (None, 5, 1, 100)         0
_________________________________________________________________
tf.compat.v1.transpose (TFOp (None, 5, 100, 1)         0
_________________________________________________________________
tf.compat.v1.squeeze (TFOpLa (None, 5, 100)            0
_________________________________________________________________
gru_1 (GRU)                  [(None, 5, 5), (None, 5)] 1605
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 30
=================================================================
Total params: 2,235
Trainable params: 2,235
Non-trainable params: 0


_________________________________________________________________
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           [(None, 5, 2)]            0
_________________________________________________________________
tf.expand_dims (TFOpLambda)  (None, 5, 2, 1)           0
_________________________________________________________________
conv2d (Conv2D)              (None, 1, 2, 100)         600
_________________________________________________________________
zero_padding2d (ZeroPadding2 (None, 5, 2, 100)         0
_________________________________________________________________
reshape_1 (Reshape)          (None, 10, 100)           0
_________________________________________________________________
gru_1 (GRU)                  [(None, 10, 5), (None, 5) 1605
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 30
=================================================================
Total params: 2,235
Trainable params: 2,235
Non-trainable params: 0
_________________________________________________________________
'''

def create_cnn(SequenceLength, regress=False):
	inputs = Input(shape=(SequenceLength, 1), name='input')
	x = Conv1D(filters=100, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
	x = MaxPooling1D()(x)
	x = BatchNormalization()(x)
	x = Flatten()(x)
	x = Dense(5)(x)
	# print(x)
	if regress:
		x = Dense(1)(x)

	# x = Conv1D(32, 3, activation='relu')(inputs)
	# x = MaxPooling1D()(x)
	# x = BatchNormalization()(x)
	# print(fc)
	model = Model(inputs, x)
	
	# print(model.summary())
	return model
'''Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           [(None, 5, 1)]            0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 5, 100)            400
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 2, 100)            0
_________________________________________________________________
batch_normalization (BatchNo (None, 2, 100)            400
_________________________________________________________________
flatten (Flatten)            (None, 200)               0
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 1005
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 1,811
Trainable params: 1,611
Non-trainable params: 200
_________________________________________________________________'''

def cnn2dLSTM(height, width, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same", activation='relu')(x)
		x = BatchNormalization()(x)
		x = Conv2D(f, (3, 3), padding="same", activation='relu')(x)
		x = BatchNormalization()(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

	x = Reshape((6*8*64,1))(x)
	x = BatchNormalization()(x)
	x = Dropout(0.25)(x)
	whole_seq_output, final_memory_state = GRU(256, return_state=True, return_sequences=True, activation='relu')(x)							
	result = Dense(5, activation='relu')(final_memory_state)
	'''# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(4)(x)
	x = Activation("relu")(x)'''

	# check to see if the regression node should be added
	if regress:
		result = Dense(1)(result)

	# construct the CNN
	model = Model(inputs, result)
	# print(model.summary())
	# return the CNN
	return model
'''Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 48, 64, 3)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 48, 64, 16)        448
_________________________________________________________________
batch_normalization (BatchNo (None, 48, 64, 16)        64
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 48, 64, 16)        2320
_________________________________________________________________
batch_normalization_1 (Batch (None, 48, 64, 16)        64
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 24, 32, 16)        0
_________________________________________________________________
dropout (Dropout)            (None, 24, 32, 16)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 32, 32)        4640
_________________________________________________________________
batch_normalization_2 (Batch (None, 24, 32, 32)        128
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 24, 32, 32)        9248
_________________________________________________________________
batch_normalization_3 (Batch (None, 24, 32, 32)        128
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 16, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 16, 32)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 12, 16, 64)        18496
_________________________________________________________________
batch_normalization_4 (Batch (None, 12, 16, 64)        256
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 12, 16, 64)        36928
_________________________________________________________________
batch_normalization_5 (Batch (None, 12, 16, 64)        256
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 8, 64)          0
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 8, 64)          0
_________________________________________________________________
reshape (Reshape)            (None, 3072, 1)           0
_________________________________________________________________
batch_normalization_6 (Batch (None, 3072, 1)           4
_________________________________________________________________
dropout_3 (Dropout)          (None, 3072, 1)           0
_________________________________________________________________
gru (GRU)                    [(None, 3072, 256), (None 198912
_________________________________________________________________
dense (Dense)                (None, 5)                 1285
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 6
=================================================================
Total params: 273,183
Trainable params: 272,733
Non-trainable params: 450
_________________________________________________________________

'''