import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling3D, MaxPooling3D, BatchNormalization, Dropout, Dense, LSTM, Reshape, Conv3D, Input, GRU
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, Input

from tensorflow.keras.models import Model

def cnnLSTM(SequenceLength, height, width, depth, regress=False):
    input_shape = (SequenceLength, height, width, 3)

    input_shape = input_shape
    inp = Input(shape=input_shape)
    # inpN = BatchNormalization()(inp)
    # pool_0 = AveragePooling3D(pool_size=(1,2,2))(inpN)
    # print(inp)
    input1 = inp[:,0,:,:,:]
    input2 = inp[:,1,:,:,:]
    input3 = inp[:,2,:,:,:]
    input4 = inp[:,3,:,:,:]
    input5 = inp[:,4,:,:,:]
    # print(input1)
    # print(input2)
    # print(input3)

    # c1 = Conv2D(16, (3, 3), activation='relu', padding="same")(input1)
    # c1 = BatchNormalization()(c1)
    # c1 = Conv2D(16, (3, 3), activation='relu', padding="same")(c1)
    # c1 = BatchNormalization()(c1)
    # c1 = MaxPooling2D(pool_size=(2,2), strides=(2, 2))(c1)
    # c1 = Conv2D(32, (3, 3), activation='relu', padding="same")(input1)
    # c1 = BatchNormalization()(c1)
    # c1 = Conv2D(32, (3, 3), activation='relu', padding="same")(c1)
    # c1 = BatchNormalization()(c1)
    # c1 = MaxPooling2D(pool_size=(2,2), strides=(2, 2))(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding="same")(input1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding="same")(c1)
    c1 = BatchNormalization()(c1)
    c1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c1)
    c1 = Conv2D(128, (3, 3), activation='relu', padding="same")(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(128, (3, 3), activation='relu', padding="same")(c1)
    c1 = BatchNormalization()(c1)
    c1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c1)
    c1 = Conv2D(256, (3, 3), activation='relu', padding="same")(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(256, (3, 3), activation='relu', padding="same")(c1)
    c1 = BatchNormalization()(c1)
    c1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c1)
    c1 = tf.reshape(c1,[-1, (c1.shape[1])*(c1.shape[2]), (c1.shape[3])])


    # c2 = Conv2D(16, (3, 3), activation='relu', padding="same")(input2)
    # c2 = BatchNormalization()(c2)
    # c2 = Conv2D(16, (3, 3), activation='relu', padding="same")(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c2)
    # c2 = Conv2D(32, (3, 3), activation='relu', padding="same")(input2)
    # c2 = BatchNormalization()(c2)
    # c2 = Conv2D(32, (3, 3), activation='relu', padding="same")(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding="same")(input2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding="same")(c2)
    c2 = BatchNormalization()(c2)
    c2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding="same")(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding="same")(c2)
    c2 = BatchNormalization()(c2)
    c2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c2)
    c2 = Conv2D(256, (3, 3), activation='relu', padding="same")(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(256, (3, 3), activation='relu', padding="same")(c2)
    c2 = BatchNormalization()(c2)
    c2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c2)
    c2 = tf.reshape(c2,[-1, (c2.shape[1])*(c2.shape[2]), (c2.shape[3])])

    
    # c3 = Conv2D(16, (3, 3), activation='relu', padding="same")(input3)
    # c3 = BatchNormalization()(c3)
    # c3 = Conv2D(16, (3, 3), activation='relu', padding="same")(c3)
    # c3 = BatchNormalization()(c3)
    # c3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c3)
    # c3 = Conv2D(32, (3, 3), activation='relu', padding="same")(input3)
    # c3 = BatchNormalization()(c3)
    # c3 = Conv2D(32, (3, 3), activation='relu', padding="same")(c3)
    # c3 = BatchNormalization()(c3)
    # c3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding="same")(input3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding="same")(c3)
    c3 = BatchNormalization()(c3)
    c3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', padding="same")(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', padding="same")(c3)
    c3 = BatchNormalization()(c3)
    c3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding="same")(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding="same")(c3)
    c3 = BatchNormalization()(c3)
    c3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c3)
    c3 = tf.reshape(c3,[-1, (c3.shape[1])*(c3.shape[2]), (c3.shape[3])])

    # c4 = Conv2D(16, (3, 3), activation='relu', padding="same")(input4)
    # c4 = BatchNormalization()(c4)
    # c4 = Conv2D(16, (3, 3), activation='relu', padding="same")(c4)
    # c4 = BatchNormalization()(c4)
    # c4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c4)
    # c4 = Conv2D(32, (3, 3), activation='relu', padding="same")(input4)
    # c4 = BatchNormalization()(c4)
    # c4 = Conv2D(32, (3, 3), activation='relu', padding="same")(c4)
    # c4 = BatchNormalization()(c4)
    # c4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding="same")(input4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding="same")(c4)
    c4 = BatchNormalization()(c4)
    c4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding="same")(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding="same")(c4)
    c4 = BatchNormalization()(c4)
    c4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', padding="same")(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', padding="same")(c4)
    c4 = BatchNormalization()(c4)
    c4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c4)
    c4 = tf.reshape(c4,[-1, (c4.shape[1])*(c4.shape[2]), (c4.shape[3])])

    # c5 = Conv2D(16, (3, 3), activation='relu', padding="same")(input5)
    # c5 = BatchNormalization()(c5)
    # c5 = Conv2D(16, (3, 3), activation='relu', padding="same")(c5)
    # c5 = BatchNormalization()(c5)
    # c5 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c5)
    # c5 = Conv2D(32, (3, 3), activation='relu', padding="same")(input5)
    # c5 = BatchNormalization()(c5)
    # c5 = Conv2D(32, (3, 3), activation='relu', padding="same")(c5)
    # c5 = BatchNormalization()(c5)
    # c5 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding="same")(input5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding="same")(c5)
    c5 = BatchNormalization()(c5)
    c5 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding="same")(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding="same")(c5)
    c5 = BatchNormalization()(c5)
    c5 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding="same")(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding="same")(c5)
    c5 = BatchNormalization()(c5)
    c5 = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(c5)
    c5 = tf.reshape(c5,[-1, (c5.shape[1])*(c5.shape[2]), (c5.shape[3])])

    # print("????????????????",c1)
    # print("????????????????",c2)

    merged = tf.concat([c1, c2, c3, c4, c5], axis=1)
    # print("????????????????",merged)
    flat = Dropout(0.2)(merged)

    whole_seq_output, final_memory_state = GRU(256, return_state=True, return_sequences=True, activation='relu')(flat)	
    out = Dense(5, activation='relu')(final_memory_state)
    # out = BatchNormalization()(out)
    # out = Dropout(0.4)(out)
    
    if regress:
        # out = Dense(1, activation='linear')(out)
        out = Dense(1)(out)
    
    model = Model(outputs=out, inputs=inp)
    # lstm_2 = GRU(5, return_sequences=True, name="gru1")(flat)
    # out = GRU(1,  return_sequences=False, name="gru2")(lstm_2)
    '''
    hidden_1 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(lstm_2)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.4)(hidden_1)
    # hidden_2 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_1)
    # hidden_2 = BatchNormalization()(hidden_2)
    # hidden_2 = Dropout(0.4)(hidden_2)
    hidden_3 = Dense(128, kernel_initializer='glorot_uniform', activation='relu')(hidden_1)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Dropout(0.4)(hidden_3)
    
    hidden_4 = Dense(8, kernel_initializer='glorot_uniform', activation='relu')(hidden_3)
    hidden_4 = BatchNormalization()(hidden_4)
    hidden_4 = Dropout(0.5)(hidden_4)
    
    out = Dense(1, activation='linear')(hidden_4)'''
    
    # model = Model(outputs=out, inputs=inp)
    
    # print(model.summary())

    return model
'''

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 5, 48, 64, 3 0
__________________________________________________________________________________________________
tf.__operators__.getitem (Slici (None, 48, 64, 3)    0           input_1[0][0]
__________________________________________________________________________________________________
tf.__operators__.getitem_1 (Sli (None, 48, 64, 3)    0           input_1[0][0]
__________________________________________________________________________________________________
tf.__operators__.getitem_2 (Sli (None, 48, 64, 3)    0           input_1[0][0]
__________________________________________________________________________________________________
tf.__operators__.getitem_3 (Sli (None, 48, 64, 3)    0           input_1[0][0]
__________________________________________________________________________________________________
tf.__operators__.getitem_4 (Sli (None, 48, 64, 3)    0           input_1[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 48, 64, 16)   448         tf.__operators__.getitem[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 48, 64, 16)   448         tf.__operators__.getitem_1[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 48, 64, 16)   448         tf.__operators__.getitem_2[0][0]
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 48, 64, 16)   448         tf.__operators__.getitem_3[0][0]
__________________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, 48, 64, 16)   448         tf.__operators__.getitem_4[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 48, 64, 16)   64          conv2d[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 48, 64, 16)   64          conv2d_10[0][0]
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 48, 64, 16)   64          conv2d_20[0][0]
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 48, 64, 16)   64          conv2d_30[0][0]
__________________________________________________________________________________________________
batch_normalization_40 (BatchNo (None, 48, 64, 16)   64          conv2d_40[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 48, 64, 16)   2320        batch_normalization[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 48, 64, 16)   2320        batch_normalization_10[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 48, 64, 16)   2320        batch_normalization_20[0][0]
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 48, 64, 16)   2320        batch_normalization_30[0][0]
__________________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, 48, 64, 16)   2320        batch_normalization_40[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 48, 64, 16)   64          conv2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 48, 64, 16)   64          conv2d_11[0][0]
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 48, 64, 16)   64          conv2d_21[0][0]
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 48, 64, 16)   64          conv2d_31[0][0]
__________________________________________________________________________________________________
batch_normalization_41 (BatchNo (None, 48, 64, 16)   64          conv2d_41[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 24, 32, 16)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 24, 32, 16)   0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
max_pooling2d_10 (MaxPooling2D) (None, 24, 32, 16)   0           batch_normalization_21[0][0]
__________________________________________________________________________________________________
max_pooling2d_15 (MaxPooling2D) (None, 24, 32, 16)   0           batch_normalization_31[0][0]
__________________________________________________________________________________________________
max_pooling2d_20 (MaxPooling2D) (None, 24, 32, 16)   0           batch_normalization_41[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 24, 32, 32)   4640        max_pooling2d[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 24, 32, 32)   4640        max_pooling2d_5[0][0]
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 24, 32, 32)   4640        max_pooling2d_10[0][0]
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 24, 32, 32)   4640        max_pooling2d_15[0][0]
__________________________________________________________________________________________________
conv2d_42 (Conv2D)              (None, 24, 32, 32)   4640        max_pooling2d_20[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 24, 32, 32)   128         conv2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 24, 32, 32)   128         conv2d_12[0][0]
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 24, 32, 32)   128         conv2d_22[0][0]
__________________________________________________________________________________________________
batch_normalization_32 (BatchNo (None, 24, 32, 32)   128         conv2d_32[0][0]
__________________________________________________________________________________________________
batch_normalization_42 (BatchNo (None, 24, 32, 32)   128         conv2d_42[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 24, 32, 32)   9248        batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 24, 32, 32)   9248        batch_normalization_12[0][0]
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 24, 32, 32)   9248        batch_normalization_22[0][0]
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 24, 32, 32)   9248        batch_normalization_32[0][0]
__________________________________________________________________________________________________
conv2d_43 (Conv2D)              (None, 24, 32, 32)   9248        batch_normalization_42[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 24, 32, 32)   128         conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 24, 32, 32)   128         conv2d_13[0][0]
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 24, 32, 32)   128         conv2d_23[0][0]
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 24, 32, 32)   128         conv2d_33[0][0]
__________________________________________________________________________________________________
batch_normalization_43 (BatchNo (None, 24, 32, 32)   128         conv2d_43[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 12, 16, 32)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 12, 16, 32)   0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
max_pooling2d_11 (MaxPooling2D) (None, 12, 16, 32)   0           batch_normalization_23[0][0]
__________________________________________________________________________________________________
max_pooling2d_16 (MaxPooling2D) (None, 12, 16, 32)   0           batch_normalization_33[0][0]
__________________________________________________________________________________________________
max_pooling2d_21 (MaxPooling2D) (None, 12, 16, 32)   0           batch_normalization_43[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 12, 16, 64)   18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 12, 16, 64)   18496       max_pooling2d_6[0][0]
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 12, 16, 64)   18496       max_pooling2d_11[0][0]
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 12, 16, 64)   18496       max_pooling2d_16[0][0]
__________________________________________________________________________________________________
conv2d_44 (Conv2D)              (None, 12, 16, 64)   18496       max_pooling2d_21[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 12, 16, 64)   256         conv2d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 12, 16, 64)   256         conv2d_14[0][0]
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 12, 16, 64)   256         conv2d_24[0][0]
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 12, 16, 64)   256         conv2d_34[0][0]
__________________________________________________________________________________________________
batch_normalization_44 (BatchNo (None, 12, 16, 64)   256         conv2d_44[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 12, 16, 64)   36928       batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 12, 16, 64)   36928       batch_normalization_14[0][0]
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 12, 16, 64)   36928       batch_normalization_24[0][0]
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 12, 16, 64)   36928       batch_normalization_34[0][0]
__________________________________________________________________________________________________
conv2d_45 (Conv2D)              (None, 12, 16, 64)   36928       batch_normalization_44[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 12, 16, 64)   256         conv2d_5[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 12, 16, 64)   256         conv2d_15[0][0]
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 12, 16, 64)   256         conv2d_25[0][0]
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 12, 16, 64)   256         conv2d_35[0][0]
__________________________________________________________________________________________________
batch_normalization_45 (BatchNo (None, 12, 16, 64)   256         conv2d_45[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 6, 8, 64)     0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 6, 8, 64)     0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
max_pooling2d_12 (MaxPooling2D) (None, 6, 8, 64)     0           batch_normalization_25[0][0]
__________________________________________________________________________________________________
max_pooling2d_17 (MaxPooling2D) (None, 6, 8, 64)     0           batch_normalization_35[0][0]
__________________________________________________________________________________________________
max_pooling2d_22 (MaxPooling2D) (None, 6, 8, 64)     0           batch_normalization_45[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 6, 8, 128)    73856       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 6, 8, 128)    73856       max_pooling2d_7[0][0]
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 6, 8, 128)    73856       max_pooling2d_12[0][0]
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 6, 8, 128)    73856       max_pooling2d_17[0][0]
__________________________________________________________________________________________________
conv2d_46 (Conv2D)              (None, 6, 8, 128)    73856       max_pooling2d_22[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 6, 8, 128)    512         conv2d_6[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 6, 8, 128)    512         conv2d_16[0][0]
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 6, 8, 128)    512         conv2d_26[0][0]
__________________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, 6, 8, 128)    512         conv2d_36[0][0]
__________________________________________________________________________________________________
batch_normalization_46 (BatchNo (None, 6, 8, 128)    512         conv2d_46[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 6, 8, 128)    147584      batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 6, 8, 128)    147584      batch_normalization_16[0][0]
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 6, 8, 128)    147584      batch_normalization_26[0][0]
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 6, 8, 128)    147584      batch_normalization_36[0][0]
__________________________________________________________________________________________________
conv2d_47 (Conv2D)              (None, 6, 8, 128)    147584      batch_normalization_46[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 6, 8, 128)    512         conv2d_7[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 6, 8, 128)    512         conv2d_17[0][0]
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 6, 8, 128)    512         conv2d_27[0][0]
__________________________________________________________________________________________________
batch_normalization_37 (BatchNo (None, 6, 8, 128)    512         conv2d_37[0][0]
__________________________________________________________________________________________________
batch_normalization_47 (BatchNo (None, 6, 8, 128)    512         conv2d_47[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 3, 4, 128)    0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
max_pooling2d_8 (MaxPooling2D)  (None, 3, 4, 128)    0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
max_pooling2d_13 (MaxPooling2D) (None, 3, 4, 128)    0           batch_normalization_27[0][0]
__________________________________________________________________________________________________
max_pooling2d_18 (MaxPooling2D) (None, 3, 4, 128)    0           batch_normalization_37[0][0]
__________________________________________________________________________________________________
max_pooling2d_23 (MaxPooling2D) (None, 3, 4, 128)    0           batch_normalization_47[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 3, 4, 256)    295168      max_pooling2d_3[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 3, 4, 256)    295168      max_pooling2d_8[0][0]
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 3, 4, 256)    295168      max_pooling2d_13[0][0]
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 3, 4, 256)    295168      max_pooling2d_18[0][0]
__________________________________________________________________________________________________
conv2d_48 (Conv2D)              (None, 3, 4, 256)    295168      max_pooling2d_23[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 3, 4, 256)    1024        conv2d_8[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 3, 4, 256)    1024        conv2d_18[0][0]
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 3, 4, 256)    1024        conv2d_28[0][0]
__________________________________________________________________________________________________
batch_normalization_38 (BatchNo (None, 3, 4, 256)    1024        conv2d_38[0][0]
__________________________________________________________________________________________________
batch_normalization_48 (BatchNo (None, 3, 4, 256)    1024        conv2d_48[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 3, 4, 256)    590080      batch_normalization_8[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 3, 4, 256)    590080      batch_normalization_18[0][0]
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 3, 4, 256)    590080      batch_normalization_28[0][0]
__________________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, 3, 4, 256)    590080      batch_normalization_38[0][0]
__________________________________________________________________________________________________
conv2d_49 (Conv2D)              (None, 3, 4, 256)    590080      batch_normalization_48[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 3, 4, 256)    1024        conv2d_9[0][0]
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 3, 4, 256)    1024        conv2d_19[0][0]
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 3, 4, 256)    1024        conv2d_29[0][0]
__________________________________________________________________________________________________
batch_normalization_39 (BatchNo (None, 3, 4, 256)    1024        conv2d_39[0][0]
__________________________________________________________________________________________________
batch_normalization_49 (BatchNo (None, 3, 4, 256)    1024        conv2d_49[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 1, 2, 256)    0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
max_pooling2d_9 (MaxPooling2D)  (None, 1, 2, 256)    0           batch_normalization_19[0][0]
__________________________________________________________________________________________________
max_pooling2d_14 (MaxPooling2D) (None, 1, 2, 256)    0           batch_normalization_29[0][0]
__________________________________________________________________________________________________
max_pooling2d_19 (MaxPooling2D) (None, 1, 2, 256)    0           batch_normalization_39[0][0]
__________________________________________________________________________________________________
max_pooling2d_24 (MaxPooling2D) (None, 1, 2, 256)    0           batch_normalization_49[0][0]
__________________________________________________________________________________________________
tf.reshape (TFOpLambda)         (None, 2, 256)       0           max_pooling2d_4[0][0]
__________________________________________________________________________________________________
tf.reshape_1 (TFOpLambda)       (None, 2, 256)       0           max_pooling2d_9[0][0]
__________________________________________________________________________________________________
tf.reshape_2 (TFOpLambda)       (None, 2, 256)       0           max_pooling2d_14[0][0]
__________________________________________________________________________________________________
tf.reshape_3 (TFOpLambda)       (None, 2, 256)       0           max_pooling2d_19[0][0]
__________________________________________________________________________________________________
tf.reshape_4 (TFOpLambda)       (None, 2, 256)       0           max_pooling2d_24[0][0]
__________________________________________________________________________________________________
tf.concat (TFOpLambda)          (None, 10, 256)      0           tf.reshape[0][0]
                                                                 tf.reshape_1[0][0]
                                                                 tf.reshape_2[0][0]
                                                                 tf.reshape_3[0][0]
                                                                 tf.reshape_4[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 10, 256)      0           tf.concat[0][0]
__________________________________________________________________________________________________
gru (GRU)                       [(None, 10, 256), (N 394752      dropout[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 5)            1285        gru[0][1]
==================================================================================================
Total params: 6,309,717
Trainable params: 6,299,797
Non-trainable params: 9,920
__________________________________________________________________________________________________


'''