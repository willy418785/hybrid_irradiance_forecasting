import tensorflow as tf
from sklearn.utils import check_array,check_consistent_length
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import logging
from pyimagesearch import parameter
import math

def VWMAPE(y_true, y_pred):
    tot = tf.reduce_sum(y_true)
    tot = tf.clip_by_value(tot, clip_value_min=1, clip_value_max=float('inf'))  # only clip tot, avoid to div 0
    vmape = tf.realdiv(tf.reduce_sum(tf.abs(tf.subtract(y_true, y_pred))), tot)  # /tot  remove *100
    return (vmape)

def corr(y_true, y_pred):
    num1 = y_true - tf.keras.backend.mean(y_true, axis=0)
    num2 = y_pred - tf.keras.backend.mean(y_pred, axis=0)
    num = tf.keras.backend.mean(num1 * num2, axis=0)
    den = tf.keras.backend.std(y_true, axis=0) * tf.keras.backend.std(y_pred, axis=0)
    return tf.keras.backend.mean(num / den)

def weighted_mean_absolute_percentage_error(y_true, y_pred): # senpai version error method
    y_true= check_array(y_true,ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    check_consistent_length(y_true, y_pred)
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)  

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true= check_array(y_true,ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    check_consistent_length(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSPE(y_true, y_pred):
    # y1 = np.array(y_pred)
    # y2 = np.array(y_true)
    # n = len(y_true)
    # temp = np.square((y1 - y2)/y2).sum()
    # score = np.sqrt(temp/n)
    y_true = check_array(y_true,ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    check_consistent_length(y_true, y_pred)
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
    return rmspe

def log_metrics(df, name):
    '''for i in range(len(df.predict)):
        if df.predict[i]<0.0:
            df.predict[i]=0
            # print(df)'''
    # df.predict[df.predict < 0.0] = 0.0
    # tdf = df[(df.ground_Truth != 0)]
    # tdf = tdf[(tdf.ground_Truth != 0)]
    # tdf.predict[tdf.predict < 0] = 0
    metircs_dist = {}
    metircs_dist["MSE"] = mean_squared_error(df[0], df[1])
    metircs_dist["RMSE"] = mean_squared_error(df[0], df[1], squared=False)
    metircs_dist["RMSPE"] = RMSPE(df[0], df[1])
    metircs_dist["MAE"] = mean_absolute_error(df[0], df[1])
    metircs_dist["MAPE"] = mean_absolute_percentage_error(df[0], df[1])
    metircs_dist["WMAPE"] = weighted_mean_absolute_percentage_error(df[0], df[1])
    gt = tf.convert_to_tensor(df[0])
    pred = tf.convert_to_tensor(df[1])
    metircs_dist["VWMAPE"] = VWMAPE(gt, pred).numpy()
    metircs_dist["corr"] = corr(gt, pred).numpy()
    log = logging.getLogger(parameter.experient_label)
    log.info("Name: {}---------------------------------------------------------".format(name))
    log.info("MSE : {}".format(metircs_dist["MSE"]))
    log.info("RMSE : {}".format(metircs_dist["RMSE"]))
    log.info("RMSPE : {}".format(metircs_dist["RMSPE"]))
    log.info("MAE : {}".format(metircs_dist["MAE"]))
    log.info("MAPE : {}".format(metircs_dist["MAPE"]))
    log.info("WMAPE : {}".format(metircs_dist["WMAPE"]))
    log.info("VWMAPE : {}".format(metircs_dist["VWMAPE"]))
    log.info("corr : {}".format(metircs_dist["corr"]))
    return metircs_dist
def seperate_log_metrics(df, name, minutes):
    '''for i in range(len(df.predict)):
        if df.predict[i]<0.0:
            df.predict[i]=0
            # print(df)'''
    # df.predict[df.predict < 0.0] = 0.0
    # tdf = df[(df.ground_Truth != 0)]
    # tdf = tdf[(df[0] != 0)]
    # df[1][df[1] < 0] = 0
    gt = df[0]['ShortWaveDown']
    pd = df[1]['ShortWaveDown']
    print(len(gt))
    glist = [[]for _ in range(minutes)]
    plist = [[]for _ in range(minutes)]

    for i in range (len(gt)):
        glist[i%minutes].append(gt[i])
        plist[i%minutes].append(pd[i])

    # print(glist)
    # print(plist)
    sep_metircs_dist = {}
    sepAvg_metircs_dist = {}
    log = logging.getLogger(parameter.experient_label)
    sep_metircs_mse = np.zeros(minutes)
    sep_metircs_rmse = np.zeros(minutes)
    sep_metircs_rmspe = np.zeros(minutes)
    sep_metircs_mae = np.zeros(minutes)
    sep_metircs_mape = np.zeros(minutes)
    sep_metircs_wmape = np.zeros(minutes)
    sep_metircs_vwmape = np.zeros(minutes)
    sep_metircs_corr = np.zeros(minutes)

    for i in range(minutes):
        sep_metircs_mse[i] = mean_squared_error(glist[i], plist[i])
        sep_metircs_rmse[i] = mean_squared_error(glist[i], plist[i], squared=False)
        sep_metircs_rmspe[i] = RMSPE(glist[i], plist[i])
        sep_metircs_mae[i] = mean_absolute_error(glist[i], plist[i])
        sep_metircs_mape[i] = mean_absolute_percentage_error(glist[i], plist[i])
        sep_metircs_wmape[i] = weighted_mean_absolute_percentage_error(glist[i], plist[i])
        gt = tf.convert_to_tensor(glist[i])
        pred = tf.convert_to_tensor(plist[i])
        sep_metircs_vwmape[i] = VWMAPE(gt, pred).numpy()
        sep_metircs_corr[i] = corr(gt, pred).numpy()

        log = logging.getLogger(parameter.experient_label)
        # log.info("Name: {}_{}---------------------------------------------------------".format(name,i+1))
        # log.info("MSE : {}".format(sep_metircs_mse[i]))
        # log.info("RMSE : {}".format(sep_metircs_rmse[i]))
        # log.info("MAE : {}".format(sep_metircs_mae[i]))
        # log.info("MAPE : {}".format(sep_metircs_mape[i]))
        # log.info("WMAPE : {}".format(sep_metircs_wmape[i]))
        log.info("{}min MSE : {}, RMSE : {}, RMSPE : {}, MAE : {}, MAPE : {}, WMAPE : {}, VWMAPE : {}, corr : {}"
                .format(i+1,sep_metircs_mse[i],sep_metircs_rmse[i],sep_metircs_rmspe[i],sep_metircs_mae[i],sep_metircs_mape[i],sep_metircs_wmape[i],sep_metircs_vwmape[i],sep_metircs_corr[i]))
        sep_metircs_dist["MSE {}min".format(i+1)] = sep_metircs_mse[i]
        sep_metircs_dist["RMSE {}min".format(i+1)] = sep_metircs_rmse[i]
        sep_metircs_dist["RMSPE {}min".format(i+1)] = sep_metircs_rmspe[i]
        sep_metircs_dist["MAE {}min".format(i+1)] = sep_metircs_mae[i]
        sep_metircs_dist["MAPE {}min".format(i+1)] = sep_metircs_mape[i]
        sep_metircs_dist["WMAPE {}min".format(i+1)] = sep_metircs_wmape[i]
        sep_metircs_dist["VWMAPE {}min".format(i+1)] = sep_metircs_vwmape[i]
        sep_metircs_dist["corr {}min".format(i+1)] = sep_metircs_corr[i]
    
    sepAvg_metircs_dist["MSE avg"] = sep_metircs_mse.mean()
    sepAvg_metircs_dist["RMSE avg"] = sep_metircs_rmse.mean()
    sepAvg_metircs_dist["RMSPE avg"] = sep_metircs_rmspe.mean()
    sepAvg_metircs_dist["MAE avg"] = sep_metircs_mae.mean()
    sepAvg_metircs_dist["MAPE avg"] = sep_metircs_mape.mean()
    sepAvg_metircs_dist["WMAPE avg"] = sep_metircs_wmape.mean()
    sepAvg_metircs_dist["VWMAPE avg"] = sep_metircs_vwmape.mean()
    sepAvg_metircs_dist["corr avg"] = sep_metircs_corr.mean()
       
    # log.info("Name: {}_avg---------------------------------------------------------".format(name))
    # log.info("avg MSE : {}".format(sepAvg_metircs_dist["MSE avg"]))
    # log.info("avg RMSE : {}".format(sepAvg_metircs_dist["RMSE avg"]))
    # log.info("avg MAE : {}".format(sepAvg_metircs_dist["MAE avg"]))
    # log.info("avg MAPE : {}".format(sepAvg_metircs_dist["MAPE avg"]))
    # log.info("avg WMAPE : {}".format(sepAvg_metircs_dist["WMAPE avg"]))
    
    log.info(sepAvg_metircs_dist)

    return sep_metircs_dist, sepAvg_metircs_dist