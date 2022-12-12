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
    # labels = ["ma_smoothed_8_4_3_transformer_numerical_weather_data_only_540_to_540_train_on_all_months", "ma_smoothed_convGRU_numerical_weather_data_only_540_to_540_train_on_all_months"]
    # labels = ["ma_smoothed_convGRU_vs_843_transformer_numerical_data_only_5_to_10"]
    # labels = ["ma_smoothed_convGRU_vs_843transformer_numerical_data_only_540_to_540_train_on_all_months"]
    # labels = ["ma_smoothed_convGRU_vs_8_4_3_transformer_numerical_weather_data_only_5_to_10", "ma_smoothed_auto_convGRU_vs_auto_8_4_3_transformer_numerical_weather_data_only_5_to_10"]
    labels = ["ma_smoothed_auto_convGRU_target_data_only_540_to_540", "ma_smoothed_unistep_convGRU_target_data_only_540_to_540", "ma_smoothed_convGRU_vs_843transformer_numerical_data_only_540_to_540_train_on_all_months"]
    experiment_name = "test"
    exclusive_models = ["Persistence", "MA"]
    # model_names = ["conv3D_c_cnnlstm", "simple_transformer"]
    rex = '$'
    # rex = '{1}.+min$'
    try:
        os.mkdir(Path("fig/ablation"))
    except:
        pass
    try:
        os.mkdir(Path("fig/ablation/{}".format(experiment_name)))
    except:
        pass
    for name in metric_names:
        all_months = pd.DataFrame()
        try:
            os.mkdir(Path("fig/ablation/{}/{}".format(experiment_name, name)))
        except:
            pass
        for month in range(1, 13):
            all = pd.DataFrame()
            mydate = datetime.datetime.strptime(str(month), "%m")
            month_name = mydate.strftime("%B")
            month_abbr = mydate.strftime("%b")
            title = "{} Test on {}".format(name, month_name)
            fig_path = "fig/ablation/{}/{}/{}".format(experiment_name, name, title)
            for label in labels:
                experiment_label = label + "_test_on_{}".format(month_abbr)
                path = "plot/{}/{}".format(experiment_label, "all_metric.csv")
                is_monthly_result_existing = os.path.exists(path)
                if not is_monthly_result_existing:
                    experiment_label = label
                    path = "plot/{}/{}".format(experiment_label, "all_metric.csv")
                try:
                    metrics = pd.read_csv(path, index_col=0)
                    metric = metrics.filter(regex='^'+ name + rex, axis='columns').round(5)
                    metric = pd.concat({label: metric}, names=['exp_name', 'models'])
                    metric.index = metric.index.swaplevel(0, 1)
                    # metric = metric.transpose()
                    # metric = metric.rename({col: "{}\n{}".format(label, col) for col in metric.columns if col not in exclusive_models}, axis=1)
                    # metric = metric.transpose()
                    all = all.append(metric)
                except Exception as ex:
                    print(str(ex))
                    break
            if len(all) > 0:
                if is_monthly_result_existing:
                    ax = plt.subplot(111)
                    unstacked = all.unstack().droplevel(0, axis=1)
                    unstacked.plot.bar(title=title, ax=ax, legend=True, table=True, figsize=(16, 9))
                    x_axis = ax.axes.get_xaxis()
                    x_axis.set_visible(False)
                    ax.legend(loc='lower right', fontsize=5)
                    # plt.show()
                    plt.savefig(fig_path)
                    plt.clf()
                all.columns = pd.MultiIndex.from_product([all.columns, [month_name]])
                all_months = pd.concat([all_months, all], axis=1)
                if not is_monthly_result_existing:
                    break
        assert len(all_months.columns) > 0
        mean = all_months.groupby(axis=1, level=0).mean().round(3).unstack().droplevel(0, axis=1)
        std_err = all_months.groupby(axis=1, level=0).sem().round(3).unstack().droplevel(0, axis=1)
        title = 'Mean {} with SE'.format(name)
        fig_path = "fig/ablation/{}/{}/{}".format(experiment_name, name, title)
        ax = plt.subplot(111)
        mean.plot.bar(yerr=std_err, capsize=3, title=title, ax=ax, legend=True, figsize=(10, 6))
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)
        ax.legend(loc='lower right', fontsize=5)
        t = (mean.astype(str) + '±' + std_err.astype(str)).transpose()
        pd.plotting.table(ax, t, loc='bottom', rowLoc='right', colLoc='center')
        # pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0+0.2*(pos.y1-pos.y0), pos.x1-pos.x0, 0.8*(pos.y1-pos.y0)])
        # plt.subplots_adjust(left=0.3, bottom=0.1)
        plt.show()
        plt.savefig(fig_path)
        plt.clf()
pass