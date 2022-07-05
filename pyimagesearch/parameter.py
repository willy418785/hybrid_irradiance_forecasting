from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# static
'''static_suffle = True
dynamic_suffle = False
timeseries = False'''
# dynamic
static_suffle = False
dynamic_suffle = False
timeseries = True
datasetPath = "../skyImage"
target = ["ShortWaveDown"]  # "ShortWaveDown","difference5","difference10"
targetAdd = "before5"  # "before5","before10"           #same number as target

dynamic_model = "one"
inputs = ["ShortWaveDown", "sun_average", "twoClass"]  # "ShortWaveDown", "twoClass", "sun_average"
addAverage = True  # True for one model:add cloud and average
# True for two model:add average

'''
1.one model:no cloud no average     -> dynamic_model = "one"  /  addAverage = False  /  inputs = ["ShortWaveDown", "sun_average", "twoClass"]
2.one model:add cloud add average   -> dynamic_model = "one"  /  addAverage = True   /  inputs = ["ShortWaveDown", "sun_average", "twoClass"]
3.two model:no average              -> dynamic_model = "two"  /  addAverage = False  /  inputs = ["ShortWaveDown", "sun_average"]
4.two model:add average             -> dynamic_model = "two"  /  addAverage = True   /  inputs = ["ShortWaveDown", "sun_average"]
'''
cloudLabel = "twoClass"
norm_mode = 2
between8_17 = True
labelScaler = MinMaxScaler()
# suffle_static_threemodel_es_cnn_resnet_solarnet    
# suffle_static_onemodel_es_cnn_resnet_solarnet_Kmeans5  
# suffle_dynamic_10_5_es_convlstm_conv3D_cnnlstm
# suffle_static_threemodel_170_220_es_cnn_resnet_solarnet
# suffle_static_onemodel_es_cnn_resnet_solarnet
# suffle_static_twomodel_200_es_cnn_resnet_solarnet
# suffle_dynamic_20_5_es_conv3D_cnnLSTM
# new_dynamic_10_5_es_conv3D_cnnLSTM_32
# dynamic_5_5_es_conv3D_cnnLSTM_MinMax
# dynamic_5_5_es_multiCnnLSTM_3Dresnet18_3Dresnet34
# dynamic_5_5_es_combined
# monthSep_dynamic_5_5_es_combined
# monthSep_dynamic_5and1_5_es_combined
# difference5_monthSep_dynamic_5and1_5_es_combined
# seq_monthSep_dynamic_0-9_14-18
experient_label = "test"  # 5x48x64x3-and-5x1-and-5x2-conv3D_c_cnnlstm    #new_twomodel_image
after_minutes = 1
input_days = None
output_days = None
input_width = 10
label_width = 10
image_input_width3D = 10
epochs = 300
# epoch_list = [100, 200, 250, 300, 400, 500]     #if no early stop
# epoch_list = [10,10,10,10,10,10]
# epoch_list = [1,1,1]
# epoch_list = [500, 500, 500, 500]
epoch_list = [500]
batchsize = 200
earlystoper = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0.0001)
# earlystoper = []

# model_list = ["data_cnn","data_cnnlstm","multiCnnLSTM_c_cnn","multiCnnLSTM_c_cnnlstm","CnnLSTM_c_cnn","CnnLSTM_c_cnnlstm"]    
# model_list = ["conv3D","cnnLSTM","multiCnnLSTM","3Dresnet18",

#             "conv3D_c_cnn",         "conv3D_c_cnnlstm",
#             "multiCnnLSTM_c_cnn",   "multiCnnLSTM_c_cnnlstm",
#             "Cnn3dLSTM_c_cnn",      "Cnn3dLSTM_c_cnnlstm",
#             "3Dresnet_c_cnn",       "3Dresnet_c_cnnlstm",

#             "data_cnnlstm", "data_cnn"] 
# model_list = ["Persistence","MA","data_cnnlstm"]
model_list = ["Persistence", "MA", "conv3D_c_cnnlstm"]
# model_list = ["Persistence","MA","conv3D_c_cnnlstm","Cnn3dLSTM_c_cnnlstm","data_cnnlstm"]
# model_list = ["Persistence","conv3D_c_cnnlstm","Cnn3dLSTM_c_cnnlstm","resnet_c_cnnlstm","solarnet_c_cnnlstm","Cnn2dLSTM_c_cnnlstm","data_cnnlstm"]
# model_list = ["Persistence","Resnet50_c_cnnlstm","Efficient_c_cnnlstm"]
# model_list = ["Persistence","conv3D_c_cnnlstm","Cnn3dLSTM_c_cnnlstm","Resnet50_c_cnnlstm","Cnn2dLSTM_c_cnnlstm","Efficient_c_cnnlstm"]

# "cnn","resnet","solarnet","convlstm","conv3D","cnn3dLSTM","multiCnnLSTM","3Dresnet","resnet2d"
# "data_cnnlstm","data_cnn"
# csvLogMetrics = ["MSE", "RMSE", "MAE", "MAPE", "WMAPE", "VWMAPE","corr"]
csvLogMetrics = ["MSE", "RMSE", "RMSPE", "MAE", "MAPE", "WMAPE", "VWMAPE", "corr",
                 "MSE 1min", "MSE 2min", "MSE 3min", "MSE 4min", "MSE 5min",
                 "RMSE 1min", "RMSE 2min", "RMSE 3min", "RMSE 4min", "RMSE 5min",
                 "RMSPE 1min", "RMSPE 2min", "RMSPE 3min", "RMSPE 4min", "RMSPE 5min",
                 "MAE 1min", "MAE 2min", "MAE 3min", "MAE 4min", "MAE 5min",
                 "MAPE 1min", "MAPE 2min", "MAPE 3min", "MAPE 4min", "MAPE 5min",
                 "WMAPE 1min", "WMAPE 2min", "WMAPE 3min", "WMAPE 4min", "WMAPE 5min",
                 "VWMAPE 1min", "VWMAPE 2min", "VWMAPE 3min", "VWMAPE 4min", "VWMAPE 5min",
                 "corr 1min", "corr 2min", "corr 3min", "corr 4min", "corr 5min",
                 "MSE avg", "RMSE avg", "RMSPE avg", "MAE avg", "MAPE avg", "WMAPE avg", "VWMAPE avg", "corr avg"]
model = "one"  # static use only
class_type = "cloud"  # static use only    # "average","cloud"
normalization = "MinMax"  # "MinMax","Mean","Standard","Max","No"
split_mode = "month"  # "all_year","month"  # all_year就是整年按比例分割dataset  # month要去改test_month:if跑main.py則只做某個月，if跑month_sep.py則test_month自己從2-8更新病最後算平均
test_month = 2
tailMonth = 8  # 最後一個月(val抓上一個月會用到)
squeeze = False  # windowgenerator裡面image要不要降成2維，用於某些model
is_using_shuffle = True
MA_width = 10

smoothing_mode = {"MA" : {"num_of_entries": 10},
                  "EMA": {"span": 10}
                  }
smoothing_type = None

is_using_sun_location = False
is_using_cloud_location = False
image_depth = 3
if is_using_sun_location:
    image_depth += 1
if is_using_cloud_location:
    image_depth += 1

timezone = 'Asia/Taipei'
