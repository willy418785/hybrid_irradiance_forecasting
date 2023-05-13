from tensorflow.keras.callbacks import EarlyStopping


class _Params:
    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        members_dict = vars(self)
        context = ""
        for k, v in members_dict.items():
            if k == 'name' or v == self.name:
                context += "{}'s parameters\n".format(str(v))
            elif issubclass(type(v), __class__):
                context += v.__str__()
            else:
                context += "\t{}: {}\n".format(str(k), str(v))
        return context


class _DataParams(_Params):
    DEFAULT_IMAGE_DEPTH = 3  # RGB image

    def __init__(self, name="Data"):
        super().__init__(name)
        # miscellaneous parameters
        self.addAverage = False  # True for one model:add cloud and average
        self.is_using_shuffle = False

        # normalization related parameters
        self.norm_mode = 'std'
        self.label_norm_mode = None

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
        self.image_depth = self.DEFAULT_IMAGE_DEPTH
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
        elif self.csv_name == 'ori_EC.csv':
            self.features = ["MT_{}".format(str(i).zfill(3)) for i in range(1, 322)]
        elif self.csv_name == '2020new.csv':
            self.features = ['ShortWaveDown']
        elif self.csv_name == 'dataset_renheo_[2019].csv':
            self.features = ["DC-1|Pdc", "DC-2|Pdc"]
        elif self.csv_name == 'dataset_renheo.csv':
            self.features = ["DC-1_Pdc", "DC-2_Pdc"]
        elif self.csv_name == "speed_index_california.csv":
            self.features = ["NorthCentralSI", "BayAreaSI", "CentralCoastSI",
                             "SouthCentralSI", "LAVenturaSI", "SanBernardinoRiversideSI",
                             "CentralSI", "SanDiegoImperialSI", "OrangeCountySI"]
        self.target = self.features

    def set_start_end_time(self):
        if self.between8_17 or self.test_between8_17:
            if self.time_granularity == 'H':
                self.start = '08:00:00'
            elif self.time_granularity == 'min' or self.time_granularity == 'T':
                self.start = '08:00:01'
            self.end = '17:00:00'

    def set_image_params(self):
        self.image_depth = self.DEFAULT_IMAGE_DEPTH
        if self.is_using_sun_location:
            self.image_depth += 1
        if self.is_using_cloud_location:
            self.image_depth += 1


class _ExpParams(_Params):
    MAX_COL_TO_PLOT = 5
    model_selection_mode = ['default', "baseline", "all", "valid", "convGRU", "transformer", "series-decomposition"]
    baselines = ["Persistence", "MA", "LR", "AR"]

    def __init__(self, name="Experiment"):
        super().__init__(name)
        self.epochs = 300
        # self.epoch_list = [100, 200, 250, 300, 400, 500]     if no early stop
        # self.epoch_list = [1]
        # self.epoch_list = [0]
        self.epoch_list = [500, 500, 500, 500, 500]
        self.batch_size = 32
        self.experiment_label = "test"
        self.callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001,
                                        restore_best_weights=True)]
        self.model_list = ["convGRU", "transformer",
                           'stationary_convGRU', "stationary_transformer",
                           'znorm_convGRU', 'znorm_transformer']
        self.save_plot = False
        self.save_csv = False

    def set_tested_models(self, mode):
        if type(mode) is int:
            if mode < 0 or mode > len(_ExpParams.model_selection_mode):
                mode = _ExpParams.model_selection_mode[0]
            else:
                mode = _ExpParams.model_selection_mode[mode]
        else:
            if mode not in _ExpParams.model_selection_mode:
                mode = _ExpParams.model_selection_mode[0]
        if mode == 'default':
            self.model_list = self.model_list
        elif mode == "baseline":
            self.model_list = _ExpParams.baselines
        elif mode == "all":
            self.model_list = ["Persistence", "MA", "LR", "AR",
                               "convGRU", "transformer",
                               'stationary_convGRU', "stationary_transformer",
                               'znorm_convGRU', 'znorm_transformer']
        elif mode == "valid":
            self.model_list = _ExpParams.baselines + ["convGRU", "transformer"]
        elif mode == "convGRU":
            self.model_list = ["convGRU", 'stationary_convGRU', 'znorm_convGRU']
        elif mode == "transformer":
            self.model_list = ["transformer", "stationary_transformer", 'znorm_transformer']
        elif mode == "series-decomposition":
            self.model_list = ['stationary_convGRU', "stationary_transformer", 'znorm_convGRU', 'znorm_transformer']
        return mode


class _ModelParams(_Params):
    def __init__(self, name="Model"):
        super().__init__(name)
        self.split_days = False
        self.bypass = "LR"
        self.time_embedding = "learnable"
        self.transformer_params = self._TransformerParams()
        self.convGRU_params = self._ConvGRUParams()
        self.bypass_params = self._BypassParams()
        self.decompose_params = self._DecomposeParams()
        self.split_day_params = self._SplitDayModuleParams()

    def set_ideal(self, dataset_name):
        self.transformer_params.set_ideal(dataset_name)
        self.convGRU_params.set_ideal(dataset_name)
        self.bypass_params.set_ideal(dataset_name)
        self.decompose_params.set_ideal(dataset_name)
        self.split_day_params.set_ideal(dataset_name)

    class _TransformerParams(_Params):
        def __init__(self, name="Transformer"):
            super().__init__(name)
            # 0.7M parameters
            self.layers = 3
            self.d_model = 64
            self.n_heads = 4
            self.dff = 256
            self.embedding_kernel_size = 7
            self.dropout_rate = 0
            self.token_length = 72

        def adjust(self, input_len):
            if input_len < self.token_length:
                self.token_length = input_len

        def set_ideal(self, dataset_name):
            if dataset_name == "EC.csv":
                pass
            elif dataset_name == "dataset_renheo.csv":
                self.layers = 4
            elif dataset_name == "speed_index_california.csv":
                self.token_length = 144

    class _ConvGRUParams(_Params):
        def __init__(self, name="ConvGRU"):
            super().__init__(name)
            # 0.7M parameters
            self.layers = 2
            self.embedding_filters = 256
            self.gru_units = 128
            self.embedding_kernel_size = 1
            self.dropout_rate = 0.1

        def set_ideal(self, dataset_name):
            if dataset_name == "EC.csv":
                pass
            elif dataset_name == "dataset_renheo.csv":
                self.dropout_rate = 0
                self.layers = 3
            elif dataset_name == "speed_index_california.csv":
                self.dropout_rate = 0.2
                self.layers = 1

    class _BypassParams(_Params):
        def __init__(self, name="Bypass"):
            super().__init__(name)
            self.order = 72

        def adjust(self, input_len):
            if input_len < self.order:
                self.order = input_len

        def set_ideal(self, dataset_name):
            if dataset_name == "EC.csv":
                pass
            elif dataset_name == "dataset_renheo.csv":
                self.order = 24
            elif dataset_name == "speed_index_california.csv":
                self.order = 120

    class _DecomposeParams(_Params):
        def __init__(self, name="Series Decomposition"):
            super().__init__(name)
            self.avg_window = 13

        def set_ideal(self, dataset_name):
            if dataset_name == "EC.csv":
                pass
            elif dataset_name == "dataset_renheo.csv":
                self.avg_window = 17
            elif dataset_name == "speed_index_california.csv":
                pass

    class _SplitDayModuleParams(_Params):
        def __init__(self, name="Split-day Module"):
            super().__init__(name)
            self.filters = 32
            self.kernel_size = 3

        def set_ideal(self, dataset_name):
            if dataset_name == "EC.csv":
                pass
            elif dataset_name == "dataset_renheo.csv":
                pass
            elif dataset_name == "speed_index_california.csv":
                pass


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

if __name__ == "__main__":
    print("######Default Configuration######")
    print(data_params)
    print(exp_params)
    print(model_params)
