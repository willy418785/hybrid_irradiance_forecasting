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
csv_name = 'EC.csv'  # ["2020final.csv","2020new.csv","2020shuffleDay", 'dataset_renheo_[2019].csv', 'EC.csv']


# features = ['ShortWaveDown'] # target only
# features = ['ShortWaveDown', 'CWB_Humidity', 'CWB_Temperature']  # david suggested
# features = ["DC-1|Pdc", "DC-2|Pdc"] # ["DC-1|Pdc", "DC-2|Pdc", "Temperature", "RH"], ["DC-1|Pdc", "DC-2|Pdc"]
# features = None
# features = ["MT_001"]
# features = ['ShortWaveDown', 'CWB_Humidity', 'CWB_WindSpeed',
#             'CWB_Temperature', 'EvapLevel', 'CWB_Rain05', 'CWB_Pressure', "CWB_WindDirection_Cosine",
#             "CWB_WindDirection_Sine"]

def set_dataset_related_params(name):
    if name == 'EC.csv':
        feat = ["MT_{}".format(str(i).zfill(3)) for i in range(1, 371)]
        # feat = ["MT_{}".format(str(i).zfill(3)) for i in range(1, 2)]
    elif name == '2020new.csv':
        feat = ['ShortWaveDown']
    elif name == 'dataset_renheo_[2019].csv':
        feat = ["DC-1|Pdc", "DC-2|Pdc"]
    tar = feat
    return feat, tar


features, target = set_dataset_related_params(csv_name)


targetAdd = "before5"  # "before5","before10"           #same number as target

dynamic_model = "one"
# inputs = ["ShortWaveDown", "twoClass", "sun_average"]
inputs = []
addAverage = False  # True for one model:add cloud and average
# True for two model:add average

'''
1.one model:no cloud no average     -> dynamic_model = "one"  /  addAverage = False  /  inputs = ["ShortWaveDown", "sun_average", "twoClass"]
2.one model:add cloud add average   -> dynamic_model = "one"  /  addAverage = True   /  inputs = ["ShortWaveDown", "sun_average", "twoClass"]
3.two model:no average              -> dynamic_model = "two"  /  addAverage = False  /  inputs = ["ShortWaveDown", "sun_average"]
4.two model:add average             -> dynamic_model = "two"  /  addAverage = True   /  inputs = ["ShortWaveDown", "sun_average"]
'''
cloudLabel = "twoClass"
norm_mode = 1
label_norm_mode = 0
time_granularity = 'H'  # 'H', 'min', 'T'
between8_17 = False
test_between8_17 = False

split_days = False
bypass = 1
time_embedding = 2

if between8_17 or test_between8_17:
    if time_granularity == 'H':
        start = '08:00:00'
    elif time_granularity == 'min' or time_granularity == 'T':
        start = '08:00:01'
    end = '17:00:00'

experiment_label = "test"
after_minutes = 1

input_width = 168
shifted_width = 0
label_width = 168
MA_width = 168
sample_rate = 24
test_sample_rate = 168

image_input_width3D = 10
is_using_image_data = False

epochs = 300
# epoch_list = [100, 200, 250, 300, 400, 500]     #if no early stop
# epoch_list = [1]
# epoch_list = [20000]
# epoch_list = [0]
# epoch_list = [500]
# epoch_list = [500, 500]
# epoch_list = [500, 500, 500]
# epoch_list = [500, 500, 500, 500]
epoch_list = [500, 500, 500, 500, 500]
batchsize = 32

earlystoper = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001,
                            restore_best_weights=True)
# earlystoper = []

# model_list = ["conv3D","cnnLSTM","multiCnnLSTM","3Dresnet18",

#             "conv3D_c_cnn",         "conv3D_c_cnnlstm",
#             "multiCnnLSTM_c_cnn",   "multiCnnLSTM_c_cnnlstm",
#             "Cnn3dLSTM_c_cnn",      "Cnn3dLSTM_c_cnnlstm",
#             "3Dresnet_c_cnn",       "3Dresnet_c_cnnlstm",

#             "data_cnnlstm", "data_cnn"]
model_list = ["Persistence", "MA",
              "convGRU", "transformer",
              'stationary_convGRU', "stationary_transformer",
              'znorm_convGRU', 'znorm_transformer']
baselines = ["Persistence", "MA"]
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
split_mode = "all_year"  # "all_year", "month", 'cross_month_validate'  # all_year就是整年按比例分割dataset  # month要去改test_month:if跑main.py則只做某個月，if跑month_sep.py則test_month自己從2-8更新病最後算平均
test_month = 2
squeeze = False  # windowgenerator裡面image要不要降成2維，用於某些model
is_using_shuffle = False

smoothing_mode = {"MA": {"num_of_entries": 10},
                  "EMA": {"span": 10}
                  }
smoothing_type = "MA"

is_using_sun_location = False
is_using_cloud_location = False
image_depth = 3  # RGB image
if is_using_sun_location:
    image_depth += 1
if is_using_cloud_location:
    image_depth += 1

timezone = 'Asia/Taipei'

save_plot = False
save_csv = False
