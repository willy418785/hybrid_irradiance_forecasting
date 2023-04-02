import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling3D, BatchNormalization, Dropout, Dense, LSTM, Flatten, Conv3D, Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

from pyimagesearch import parameter


def conv3D(SequenceLength, height, width, depth = parameter.data_params.image_depth, predict_length=parameter.data_params.label_width, regress=False):
    input_shape = (SequenceLength, height, width, depth)

    inp = Input(shape=input_shape)
    inpN = BatchNormalization()(inp)
    input_last = inp[:,-1,:,:,:]
    
    c1 = Conv3D(filters=16, kernel_size= (SequenceLength,5,5), strides=(1,1,1), activation='relu', padding='same')(inpN)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(filters=16, kernel_size= (SequenceLength,5,5), strides=(1,1,1), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = MaxPooling3D(pool_size=(1,2,2))(c1)
    drop_1 = Dropout(0.25)(c1)
    
    c2 = Conv3D(filters=32, kernel_size= (SequenceLength,3,3), strides=(1,1,1), activation='relu', padding='same')(drop_1)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(filters=32, kernel_size= (SequenceLength,3,3), strides=(1,1,1), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = MaxPooling3D(pool_size=(1,2,2))(c2)
    drop_2 = Dropout(0.25)(c2)
    
    c3 = Conv3D(filters=64, kernel_size= (SequenceLength,3,3), strides=(1,1,1), activation='relu', padding='same')(drop_2)
    c3 = BatchNormalization()(c3)
    c3 = Conv3D(filters=64, kernel_size= (SequenceLength,3,3), strides=(1,1,1), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = MaxPooling3D(pool_size=(1,2,2))(c3)
    drop_3 = Dropout(0.25)(c3)

    c4 = Conv3D(filters=128, kernel_size= (SequenceLength,3,3), strides=(1,1,1), activation='relu', padding='same')(drop_3)
    c4 = BatchNormalization()(c4)
    c4 = Conv3D(filters=128, kernel_size= (SequenceLength,3,3), strides=(1,1,1), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = MaxPooling3D(pool_size=(1,2,2))(c4)
    drop_4 = Dropout(0.25)(c4)

    c5 = Conv3D(filters=256, kernel_size= (SequenceLength,3,3), strides=(1,1,1), activation='relu', padding='same')(drop_4)
    c5 = BatchNormalization()(c5)
    c5 = Conv3D(filters=256, kernel_size= (SequenceLength,3,3), strides=(1,1,1), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = MaxPooling3D(pool_size=(1,2,2))(c5)
    drop_5 = Dropout(0.25)(c5)
    flat = Flatten()(drop_5)
    

    o5 = MaxPooling2D(pool_size=(2,2))(input_last)
    o5 = Conv2D(32, (5, 3), activation='relu', padding="same")(o5)
    o5 = BatchNormalization()(o5)
    o5 = Conv2D(32, (5, 3), activation='relu', padding="same")(o5)
    o5 = BatchNormalization()(o5)
    o5 = MaxPooling2D(pool_size=(2,2))(o5)
    o5 = Conv2D(64, (5, 3), activation='relu', padding="same")(o5)
    o5 = BatchNormalization()(o5)
    o5 = Conv2D(64, (5, 3), activation='relu', padding="same")(o5)
    o5 = BatchNormalization()(o5)
    o5 = MaxPooling2D(pool_size=(2,2))(o5)
    o5 = Conv2D(128, (5, 3), activation='relu', padding="same")(o5)
    o5 = BatchNormalization()(o5)
    o5 = Conv2D(128, (5, 3), activation='relu', padding="same")(o5)
    o5 = BatchNormalization()(o5)
    o5 = MaxPooling2D(pool_size=(2,2))(o5)
    o5 = Conv2D(256, (5, 3), activation='relu', padding="same")(o5)
    o5 = BatchNormalization()(o5)
    o5 = Conv2D(256, (5, 3), activation='relu', padding="same")(o5)
    o5 = BatchNormalization()(o5)
    o5 = MaxPooling2D(pool_size=(2,2))(o5)
    o5 = Flatten()(o5)
    print(o5)
    merged = tf.concat([flat,o5], axis=1)

    out = Dense(256, activation='relu')(merged)
    # out = BatchNormalization()(out)
    # out = Dropout(0.4)(out)
    out = Dense(predict_length, activation='relu')(out)
    # out = BatchNormalization()(out)
    # out = Dropout(0.4)(out)
    if regress:
        # out = Dense(1, activation='linear')(out)
        out = Dense(1)(out)
        
    out = tf.expand_dims(out, axis=-1)
    model = Model(outputs=out, inputs=inp)
    
    # print(model.summary())

    return model

'''def conv3D(SequenceLength, height, width, depth, regress=False):
    input_shape = (SequenceLength, height, width, 3)

    inp = Input(shape=input_shape)
    inpN = BatchNormalization()(inp)
    
    c1 = Conv3D(filters=16, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(inpN)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(filters=16, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    pool_1 = MaxPooling3D(pool_size=(1,2,2))(c1)
    drop_1 = Dropout(0.25)(pool_1)
    
    c2 = Conv3D(filters=32, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(drop_1)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(filters=32, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    pool_2 = MaxPooling3D(pool_size=(1,2,2))(c2)
    drop_2 = Dropout(0.25)(pool_2)
    
    c3 = Conv3D(filters=64, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(drop_2)
    c3 = BatchNormalization()(c3)
    c3 = Conv3D(filters=64, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    pool_3 = MaxPooling3D(pool_size=(1,2,2))(c3)
    drop_3 = Dropout(0.25)(pool_3)

    c4 = Conv3D(filters=128, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(drop_3)
    c4 = BatchNormalization()(c4)
    c4 = Conv3D(filters=128, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    pool_4 = MaxPooling3D(pool_size=(1,2,2))(c4)
    drop_4 = Dropout(0.25)(pool_4)

    c5 = Conv3D(filters=256, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(drop_4)
    c5 = BatchNormalization()(c5)
    c5 = Conv3D(filters=256, kernel_size= (3,3,3), strides=(1,1,1), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    pool_5 = MaxPooling3D(pool_size=(1,2,2))(c5)
    drop_5 = Dropout(0.25)(pool_5)

    flat = Flatten()(drop_5)
    out = Dense(256)(flat)
    # out = BatchNormalization()(out)
    # out = Dropout(0.4)(out)
    out = Dense(5)(out)
    # out = BatchNormalization()(out)
    # out = Dropout(0.4)(out)
    
    if regress:
        # out = Dense(1, activation='linear')(out)
        out = Dense(1)(out)
    
    model = Model(outputs=out, inputs=inp)
    
    # print(model.summary())

    return model
'''

'''Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5, 48, 64, 3)]    0
_________________________________________________________________
batch_normalization (BatchNo (None, 5, 48, 64, 3)      12
_________________________________________________________________
conv3d (Conv3D)              (None, 5, 48, 64, 16)     1312
_________________________________________________________________
batch_normalization_1 (Batch (None, 5, 48, 64, 16)     64
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 5, 48, 64, 16)     6928
_________________________________________________________________
batch_normalization_2 (Batch (None, 5, 48, 64, 16)     64
_________________________________________________________________
max_pooling3d (MaxPooling3D) (None, 5, 24, 32, 16)     0
_________________________________________________________________
dropout (Dropout)            (None, 5, 24, 32, 16)     0
_________________________________________________________________
conv3d_2 (Conv3D)            (None, 5, 24, 32, 32)     13856
_________________________________________________________________
batch_normalization_3 (Batch (None, 5, 24, 32, 32)     128
_________________________________________________________________
conv3d_3 (Conv3D)            (None, 5, 24, 32, 32)     27680
_________________________________________________________________
batch_normalization_4 (Batch (None, 5, 24, 32, 32)     128
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 5, 12, 16, 32)     0
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 12, 16, 32)     0
_________________________________________________________________
conv3d_4 (Conv3D)            (None, 5, 12, 16, 64)     55360
_________________________________________________________________
batch_normalization_5 (Batch (None, 5, 12, 16, 64)     256
_________________________________________________________________
conv3d_5 (Conv3D)            (None, 5, 12, 16, 64)     110656
_________________________________________________________________
batch_normalization_6 (Batch (None, 5, 12, 16, 64)     256
_________________________________________________________________
max_pooling3d_2 (MaxPooling3 (None, 5, 6, 8, 64)       0
_________________________________________________________________
dropout_2 (Dropout)          (None, 5, 6, 8, 64)       0
_________________________________________________________________
conv3d_6 (Conv3D)            (None, 5, 6, 8, 128)      221312
_________________________________________________________________
batch_normalization_7 (Batch (None, 5, 6, 8, 128)      512
_________________________________________________________________
conv3d_7 (Conv3D)            (None, 5, 6, 8, 128)      442496
_________________________________________________________________
batch_normalization_8 (Batch (None, 5, 6, 8, 128)      512
_________________________________________________________________
max_pooling3d_3 (MaxPooling3 (None, 5, 3, 4, 128)      0
_________________________________________________________________
dropout_3 (Dropout)          (None, 5, 3, 4, 128)      0
_________________________________________________________________
conv3d_8 (Conv3D)            (None, 5, 3, 4, 256)      884992
_________________________________________________________________
batch_normalization_9 (Batch (None, 5, 3, 4, 256)      1024
_________________________________________________________________
conv3d_9 (Conv3D)            (None, 5, 3, 4, 256)      1769728
_________________________________________________________________
batch_normalization_10 (Batc (None, 5, 3, 4, 256)      1024
_________________________________________________________________
max_pooling3d_4 (MaxPooling3 (None, 5, 1, 2, 256)      0
_________________________________________________________________
dropout_4 (Dropout)          (None, 5, 1, 2, 256)      0
_________________________________________________________________
flatten (Flatten)            (None, 2560)              0
_________________________________________________________________
dense (Dense)                (None, 256)               655616
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 1285
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 4,195,207
Trainable params: 4,193,217
Non-trainable params: 1,990
_________________________________________________________________


'''

'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 5, 48, 64, 3 0
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 5, 48, 64, 3) 12          input_1[0][0]
__________________________________________________________________________________________________
conv3d (Conv3D)                 (None, 5, 48, 64, 16 6016        batch_normalization[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 5, 48, 64, 16 64          conv3d[0][0]
__________________________________________________________________________________________________
conv3d_1 (Conv3D)               (None, 5, 48, 64, 16 32016       batch_normalization_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 5, 48, 64, 16 64          conv3d_1[0][0]
__________________________________________________________________________________________________
max_pooling3d (MaxPooling3D)    (None, 5, 24, 32, 16 0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 5, 24, 32, 16 0           max_pooling3d[0][0]
__________________________________________________________________________________________________
conv3d_2 (Conv3D)               (None, 5, 24, 32, 32 23072       dropout[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 5, 24, 32, 32 128         conv3d_2[0][0]
__________________________________________________________________________________________________
conv3d_3 (Conv3D)               (None, 5, 24, 32, 32 46112       batch_normalization_3[0][0]
__________________________________________________________________________________________________
tf.__operators__.getitem (Slici (None, 48, 64, 3)    0           input_1[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 5, 24, 32, 32 128         conv3d_3[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 24, 32, 3)    0           tf.__operators__.getitem[0][0]
__________________________________________________________________________________________________
max_pooling3d_1 (MaxPooling3D)  (None, 5, 12, 16, 32 0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 24, 32, 32)   896         max_pooling2d[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 5, 12, 16, 32 0           max_pooling3d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 24, 32, 32)   128         conv2d[0][0]
__________________________________________________________________________________________________
conv3d_4 (Conv3D)               (None, 5, 12, 16, 64 92224       dropout_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 24, 32, 32)   9248        batch_normalization_11[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 5, 12, 16, 64 256         conv3d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 24, 32, 32)   128         conv2d_1[0][0]
__________________________________________________________________________________________________
conv3d_5 (Conv3D)               (None, 5, 12, 16, 64 184384      batch_normalization_5[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 12, 16, 32)   0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 5, 12, 16, 64 256         conv3d_5[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 12, 16, 64)   18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
max_pooling3d_2 (MaxPooling3D)  (None, 5, 6, 8, 64)  0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 12, 16, 64)   256         conv2d_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 5, 6, 8, 64)  0           max_pooling3d_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 12, 16, 64)   36928       batch_normalization_13[0][0]
__________________________________________________________________________________________________
conv3d_6 (Conv3D)               (None, 5, 6, 8, 128) 368768      dropout_2[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 12, 16, 64)   256         conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 5, 6, 8, 128) 512         conv3d_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 6, 8, 64)     0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
conv3d_7 (Conv3D)               (None, 5, 6, 8, 128) 737408      batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 6, 8, 128)    73856       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 5, 6, 8, 128) 512         conv3d_7[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 6, 8, 128)    512         conv2d_4[0][0]
__________________________________________________________________________________________________
max_pooling3d_3 (MaxPooling3D)  (None, 5, 3, 4, 128) 0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 6, 8, 128)    147584      batch_normalization_15[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 5, 3, 4, 128) 0           max_pooling3d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 6, 8, 128)    512         conv2d_5[0][0]
__________________________________________________________________________________________________
conv3d_8 (Conv3D)               (None, 5, 3, 4, 256) 1474816     dropout_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 3, 4, 128)    0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 5, 3, 4, 256) 1024        conv3d_8[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 3, 4, 256)    295168      max_pooling2d_3[0][0]
__________________________________________________________________________________________________
conv3d_9 (Conv3D)               (None, 5, 3, 4, 256) 2949376     batch_normalization_9[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 3, 4, 256)    1024        conv2d_6[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 5, 3, 4, 256) 1024        conv3d_9[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 3, 4, 256)    590080      batch_normalization_17[0][0]
__________________________________________________________________________________________________
max_pooling3d_4 (MaxPooling3D)  (None, 5, 1, 2, 256) 0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 3, 4, 256)    1024        conv2d_7[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 5, 1, 2, 256) 0           max_pooling3d_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 1, 2, 256)    0           batch_normalization_18[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 2560)         0           dropout_4[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 512)          0           max_pooling2d_4[0][0]
__________________________________________________________________________________________________
tf.concat (TFOpLambda)          (None, 3072)         0           flatten[0][0]
                                                                 flatten_1[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          786688      tf.concat[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 5)            1285        dense[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            6           dense_1[0][0]
==================================================================================================
Total params: 7,882,247
Trainable params: 7,878,337
Non-trainable params: 3,910
__________________________________________________________________________________________________
'''