import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime
from pathlib import Path

'''
Plot performance bar chart of multiple experiments for visual comparison.
Note that all tested models should be included after looking up every specified all_metrics.csv, otherwise, it would get assertion failure.
-------------------------------------------------------------------------
metric_names:    list<str>, contains names of metrics that is desired to be plotted
labels:          list<str>, contains dir names in ./plot that have all_metrics.csv located
experiment_name: str, the name of dir in ./fig where all resulting figures belong
rex:             str, regular expression for filtering wanted metrics
'''

if __name__ == '__main__':
    metric_names = ["MSE", "RMSE", "RMSPE", "MAE", "MAPE", "WMAPE", "VWMAPE", "corr"]
    # labels = ["8to15_hourly_day2day_david_suggested_weather_data"]
    # labels = ["ma_smoothed_auto_vs_unistep_convGRU_vs_transformer_target_data_only_5_to_10"]
    # labels = ["ma_smoothed_auto_vs_unistep_convGRU_vs_transformer_david_suggested_weather_data_only_5_to_10"]
    # labels = ["ma_smoothed_auto_convGRU_vs_auto_8_4_3_transformer_numerical_weather_data_only_5_to_10", "ma_smoothed_convGRU_vs_8_4_3_transformer_numerical_weather_data_only_5_to_10"]
    # labels = ['ma_smoothed_8_4_3_transformer_numerical_weather_data_only_540_to_540_train_on_all_months', 'ma_smoothed_convGRU_numerical_weather_data_only_540_to_540_train_on_all_months']
    # labels = ['ma_smoothed_unistep_transformer_target_data_only_540_to_540',
    #           'ma_smoothed_unistep_convGRU_target_data_only_540_to_540']
    labels = ['ma_smoothed_transformer_david_suggested_weather_data_only_540_to_540',
              'ma_smoothed_convGRU_david_suggested_weather_data_only_540_to_540']
    # labels = ["8to15_hourly_day2day_renheo[2019]_david_suggested_weather_data"]
    # experiment_name = "david_suggested_weather_data_only_5_to_10"
    # experiment_name = "ma_smoothed_weather_data_only_540_to_540"
    # experiment_name = "ma_smoothed_target_data_only_540_to_540"
    experiment_name = "ma_smoothed_david_suggested_weather_data_only_540_to_540"
    # experiment_name = '8to15_hourly_day2day_renheo[2019]_david_suggested_weather_data_hours_separated'
    # experiment_name = '8to15_hourly_day2day_david_suggested_weather_data_hours_separated'
    # model_names = ["Persistence", "MA", "datamodel_CL", "autoregressive_convGRU", "simple_transformer", "autoregressive_transformer"]
    # model_names = ["Persistence", "MA", "convGRU", "autoregressive_convGRU", "simple_transformer",
    #                "autoregressive_transformer"]
    model_names = ["Persistence", "MA", "datamodel_CL", "simple_transformer"]  # tested models
    rex = '$'
    # rex = '{1}.+min$'
    try:
        os.mkdir(Path("fig"))
    except:
        pass
    try:
        os.mkdir(Path("fig/{}".format(experiment_name)))
    except:
        pass
    for name in metric_names:
        total_months = 0
        mean = None
        try:
            os.mkdir(Path("fig/{}/{}".format(experiment_name, name)))
        except:
            pass
        for month in range(1, 9):
            all = pd.DataFrame()
            total_months += 1
            mydate = datetime.datetime.strptime(str(month), "%m")
            month_name = mydate.strftime("%B")
            month_abbr = mydate.strftime("%b")
            title = "Test on {}".format(month_name)
            fig_path = "fig/{}/{}/{}".format(experiment_name, name, title)
            for label in labels:
                experiment_label = label + "_test_on_{}".format(month_abbr)
                path = "plot/{}/{}".format(experiment_label, "all_metric.csv")
                try:
                    metrics = pd.read_csv(path, index_col=0)
                    metric = metrics.filter(regex='^' + name + rex, axis='columns').round(5)
                    models = list((set(model_names) & set(list(metric.index))) - set(all.index))
                    models = [ele for ele in model_names if ele in models]
                    metric = metric.loc[models]
                    all = all.append(metric)
                except Exception as ex:
                    print(str(ex))
                    if type(ex) is FileNotFoundError:
                        total_months -= 1
                        break
            if len(all) == len(model_names):
                all = all.reindex(model_names)
                ax = plt.subplot(211)
                all.plot.bar(title=title, ax=ax, legend=True, table=True, figsize=(10, 6), layout="constrained")
                x_axis = ax.axes.get_xaxis()
                x_axis.set_visible(False)
                ax.legend(loc='lower right', fontsize=5)
                plt.savefig(fig_path)
                plt.clf()
                if mean is None:
                    mean = all
                else:
                    mean = mean.add(all)
        assert mean is not None
        mean = mean.div(total_months).round(5)
        title = 'Mean'
        fig_path = "fig/{}/{}/{}".format(experiment_name, name, title)
        ax = plt.subplot(211)
        mean.plot.bar(title=title, ax=ax, legend=True, table=True, figsize=(10, 6), layout="constrained")
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)
        ax.legend(loc='lower right', fontsize=5)
        plt.savefig(fig_path)
        plt.clf()
