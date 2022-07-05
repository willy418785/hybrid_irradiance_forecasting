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
    labels = ["15_to_10"]
    for label in labels:
        try:
            os.mkdir(Path("fig/{}".format(label)))
        except:
            print("fig Dir exist")
        for name in metric_names:
            mean = None
            try:
                os.mkdir(Path("fig/{}/{}".format(label, name)))
            except:
                print("fig Dir exist")
            for month in range(3, 9):
                total_months += 1
                mydate = datetime.datetime.strptime(str(month), "%m")
                month_name = mydate.strftime("%B")
                month_abbr = mydate.strftime("%b")
                title = "Test on {}".format(month_name)
                experiment_label = label + "_test_on_{}".format(month_abbr)
                # experiment_label = ("{}_test_" + label).format(month)
                path = "plot/{}/{}".format(experiment_label, "all_metric.csv")
                fig_path = "fig/{}/{}/{}".format(label, name, title)
                try:
                    metrics = pd.read_csv(path, index_col=0)
                    metric = metrics.filter(regex='^'+ name +'{1}.+min$', axis='columns').round(5)
                    if mean is None:
                        mean = metric
                    else:
                        mean = mean.add(metric)
                    ax = plt.subplot(211)
                    metric.plot.bar(title=title, ax=ax, legend=True, table=True, figsize=(10, 6))
                    x_axis = ax.axes.get_xaxis()
                    x_axis.set_visible(False)
                    ax.legend(loc='lower right', fontsize=5)
                    plt.savefig(fig_path)
                    plt.clf()
                except:
                    print("Path \"" + path + "\" doesn't exist")
            mean = mean.div(total_months).round(5)
            title = 'Mean'
            fig_path = "fig/{}/{}/{}".format(label, name, title)
            ax = plt.subplot(211)
            mean.plot.bar(title=title, ax=ax, legend=True, table=True, figsize=(10, 6))
            x_axis = ax.axes.get_xaxis()
            x_axis.set_visible(False)
            ax.legend(loc='lower right', fontsize=5)
            plt.savefig(fig_path)
            plt.clf()
    pass