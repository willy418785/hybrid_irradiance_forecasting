from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


class _DataParams:
    def __init__(self):
        # miscellaneous parameters
        self.addAverage = False  # True for one model:add cloud and average
        self.is_using_shuffle = False

        # normalization related parameters
        self.norm_mode = 1
        self.label_norm_mode = 0

        # parameters that used to specify different dataset
        self.csv_name = 'EC.csv'
        self.features = None
        self.target = None

        # data sample structure related parameters
        self.time_granularity = 'H'  # 'H', 'min', 'T'
        self.between8_17 = False
        self.test_between8_17 = False
        self.start = None
        self.end = None

        # input/output formation defining parameters
        self.input_width = 168
        self.shifted_width = 0
        self.label_width = 168
        self.MA_width = 168
        self.sample_rate = 24
        self.test_sample_rate = 168

        # data splitting related parameters
        self.split_mode = "all_year"  # ["all_year", "month", 'cross_month_validate']
        self.test_month = 2

        # target smoothing related parameters
        self.smoothing_type = None
        self.smoothing_parameter = None

        # images related parameters
        self.image_path = "../skyImage"
        self.is_using_image_data = False
        self.image_input_width3D = 10
        self.is_using_sun_location = False
        self.is_using_cloud_location = False
        self.image_depth = 3  # default RGB image
        self.timezone = 'Asia/Taipei'
        self.squeeze = False  # windowgenerator裡面image要不要降成2維，用於某些model

        # update dynamic parameters
        self.set_dataset_params()
        self.set_start_end_time()
        self.set_image_params()

    def set_dataset_params(self):
        # self.features = ['ShortWaveDown'] # target only
        # self.features = ['ShortWaveDown', 'CWB_Humidity', 'CWB_Temperature']  # david suggested
        # self.features = ["DC-1|Pdc", "DC-2|Pdc"] # ["DC-1|Pdc", "DC-2|Pdc", "Temperature", "RH"], ["DC-1|Pdc", "DC-2|Pdc"]
        # self.features = ['ShortWaveDown', 'CWB_Humidity', 'CWB_WindSpeed',
        #                  'CWB_Temperature', 'EvapLevel', 'CWB_Rain05', 'CWB_Pressure', "CWB_WindDirection_Cosine",
        #                  "CWB_WindDirection_Sine"]
        if self.csv_name == 'EC.csv':
            self.features = ["MT_{}".format(str(i).zfill(3)) for i in range(1, 371)]
            # feat = ["MT_{}".format(str(i).zfill(3)) for i in range(1, 2)]
        elif self.csv_name == '2020new.csv':
            self.features = ['ShortWaveDown']
        elif self.csv_name == 'dataset_renheo_[2019].csv':
            self.features = ["DC-1|Pdc", "DC-2|Pdc"]
        self.target = self.features

    def set_start_end_time(self):
        if self.between8_17 or self.test_between8_17:
            if self.time_granularity == 'H':
                self.start = '08:00:00'
            elif self.time_granularity == 'min' or self.time_granularity == 'T':
                self.start = '08:00:01'
            self.end = '17:00:00'

    def set_image_params(self):
        self.image_depth = 3
        if self.is_using_sun_location:
            self.image_depth += 1
        if self.is_using_cloud_location:
            self.image_depth += 1


class _ExpParams:
    def __init__(self):
        self.epochs = 300
        # self.epoch_list = [100, 200, 250, 300, 400, 500]     if no early stop
        self.epoch_list = [1]
        # self.epoch_list = [0]
        # self.epoch_list = [500, 500, 500, 500, 500]
        self.batch_size = 32
        self.experiment_label = "test"
        self.callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001,
                                        restore_best_weights=True)]
        self.model_list = ["Persistence", "MA",
                           "convGRU", "transformer",
                           'stationary_convGRU', "stationary_transformer",
                           'znorm_convGRU', 'znorm_transformer']
        self.baselines = ["Persistence", "MA"]
        self.save_plot = False
        self.save_csv = False


class _ModelParams:
    def __init__(self):
        self.split_days = False
        self.bypass = 1
        self.time_embedding = 2
        self.transformer_params = self._TransformerParams()
        self.convGRU_params = self._ConvGRUParams()
        self.bypass_params = self._BypassParams()
        self.decompose_params = self._DecomposeParams()
        self.split_day_params = self._SplitDayModuleParams()

    class _TransformerParams:
        def __init__(self):
            self.layers = 1
            self.d_model = 32
            self.n_heads = 1
            self.dff = 128
            self.embedding_kernel_size = 3
            self.dropout_rate = 0.1

    class _ConvGRUParams:
        def __init__(self):
            self.layers = 1
            self.embedding_filters = 32
            self.gru_units = 32
            self.embedding_kernel_size = 3
            self.dropout_rate = 0.1

    class _BypassParams:
        def __init__(self):
            self.order = 24

    class _DecomposeParams:
        def __init__(self):
            self.avg_window = 17

    class _SplitDayModuleParams:
        def __init__(self):
            self.filters = 32
            self.kernel_size = 3


data_params = _DataParams()
exp_params = _ExpParams()
model_params = _ModelParams()
# dynamic
static_suffle = False
dynamic_suffle = False

targetAdd = "before5"  # "before5","before10"           #same number as target

dynamic_model = "one"
# inputs = ["ShortWaveDown", "twoClass", "sun_average"]
inputs = []

'''
1.one model:no cloud no average     -> dynamic_model = "one"  /  addAverage = False  /  inputs = ["ShortWaveDown", "sun_average", "twoClass"]
2.one model:add cloud add average   -> dynamic_model = "one"  /  addAverage = True   /  inputs = ["ShortWaveDown", "sun_average", "twoClass"]
3.two model:no average              -> dynamic_model = "two"  /  addAverage = False  /  inputs = ["ShortWaveDown", "sun_average"]
4.two model:add average             -> dynamic_model = "two"  /  addAverage = True   /  inputs = ["ShortWaveDown", "sun_average"]
'''
cloudLabel = "twoClass"
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

normalization = "MinMax"  # "MinMax","Mean","Standard","Max","No"
