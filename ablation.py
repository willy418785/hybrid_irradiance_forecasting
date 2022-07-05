import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime
from pathlib import Path

if __name__ == '__main__':
    total_months = 0
    metric_names = ["MSE", "RMSE", "RMSPE", "MAE", "MAPE", "WMAPE", "VWMAPE", "corr"]
    labels = ["shuffled_5_to_10", "ma_smoothed_5_to_10", "ema_smoothed_5_to_10"]
    labels = ["shuffled_5_to_10", "5_image_10_data_10_predict", "10_to_10", "15_to_10"]
    experiment_name = "data_smoothing"
    experiment_name = "input_length"
    model_name = "conv3D_c_cnnlstm"
    try:
        os.mkdir(Path("fig/ablation"))
    except:
        pass
    try:
        os.mkdir(Path("fig/ablation/{}".format(experiment_name)))
    except:
        pass
    for name in metric_names:
        mean = None
        try:
            os.mkdir(Path("fig/ablation/{}/{}".format(experiment_name, name)))
        except:
            pass
        for month in range(3, 9):
            all = pd.DataFrame()
            total_months += 1
            mydate = datetime.datetime.strptime(str(month), "%m")
            month_name = mydate.strftime("%B")
            month_abbr = mydate.strftime("%b")
            title = "Test on {}".format(month_name)
            fig_path = "fig/ablation/{}/{}/{}".format(experiment_name, name, title)
            for label in labels:
                experiment_label = label + "_test_on_{}".format(month_abbr)
                path = "plot/{}/{}".format(experiment_label, "all_metric.csv")
                try:
                    metrics = pd.read_csv(path, index_col=0)
                    metric = metrics.filter(regex='^'+ name +'{1}.+min$', axis='columns').round(5)
                    metric = metric.loc[model_name]
                    metric = metric.rename(label)
                    all = all.append(metric)
                except:
                    print("Path \"" + path + "\" doesn't exist")

            ax = plt.subplot(211)
            all.plot.bar(title=title, ax=ax, legend=True, table=True, figsize=(10, 6))
            x_axis = ax.axes.get_xaxis()
            x_axis.set_visible(False)
            ax.legend(loc='lower right', fontsize=5)
            plt.savefig(fig_path)
            plt.clf()
            if mean is None:
                mean = all
            else:
                mean = mean.add(all)
        mean = mean.div(total_months).round(5)
        title = 'Mean'
        fig_path = "fig/ablation/{}/{}/{}".format(experiment_name, name, title)
        ax = plt.subplot(211)
        mean.plot.bar(title=title, ax=ax, legend=True, table=True, figsize=(10, 6))
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)
        ax.legend(loc='lower right', fontsize=5)
        plt.savefig(fig_path)
        plt.clf()
pass