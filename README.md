# DL Framework for Multivariate Periodic Time Series Forecasting

## Abstract

Keras implementation of proposed DL framework for general multivariate periodic time series forecasting.

The whole framework can be divided into 3 parts:

1. Input representation
    - Extract short-term patterns and datetime information from raw input sequences
2. Encoder-Decoder sequence model
    - Model long-term temporal dependencies in input sequences
3. Bypass linear unit
    - Enhance local responses of model's predictions

The overall computation flow is shown in the following image
<p align="center">
  <img src="img\model.png">
</p>

## Methodology

Every component in our proposed framework has multiple variants and can be customized by
modifying `pyimagesearch/parameter.py` or passing arguments to `main.py`

### Datetime embeddings

We support 4 types of embedding method:

* None
    * Do not use datetime embeddings
* Time2Vec
    * https://arxiv.org/abs/1907.05321
* Learnable (default)
    * Learn multiple flexible and independent linear embeddings for different time granularity
* Sine & Cosine
    * Map datetime data to [0, 1] via sine and cosine functions

### Models

We support 2 encoder-decoder seq. models:

* GRU
* Transformer

### Linear Unit

We support 3 types of bypass linear unit:

* None
    * Use nothing
* Temporal linear regression
    * Prediction of each step is a linear combination of past observations
* Periodic Moving Average
    * Average over historical values for each hour over multiple past days

## Getting Started

### Prerequisites

Python version >= 3.8.10

The listed libraries below are mandatory

* Tensorflow 2.9.1
* scikit-learn 1.1.2
* pandas 1.4.3
* plotly 5.10.0
* opencv-python 4.6.0 (Mostly unused)
* pvlib 0.9.1 (Mostly unused)
* matplotlib 3.5.1 (Mostly unused)

### Datasets

Currently, only 3 Datasets are supported:

* `ori_EC.csv`
* `speed_index_califorina,csv`
* `dataset_renheo.csv`

You can download these datasets at https://github.com/willy418785/time_series_datasets.git in csv formation. 
Place them under the root directory of this repository.

### Run Main Program

Full training and testing logic reside in python file `main.py`

For example, user can run the main program under default configuration as follows

```shell
python main.py -n "trial_0"
# -n is a required argument followed by the custom name ("trial_0" in this case) of this program run activity
```

For detailed explanations of all arguments, sepecify `-h` flag as follows

```shell
python main.py -h
```

User can change running configuration via two approaches
* Passing arguments when running `main.py` **(recommended)**
* Directly modify configuration parameters in `pyimagesearch/parameter.py`

