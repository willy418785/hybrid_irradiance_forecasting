import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime
from pathlib import Path

'''
Plot performance bar chart of multiple experiments for visual comparison.
-------------------------------------------------------------------------
metric_names:    list<str>, contains names of metrics that is desired to be plotted
models:          list<str>, list of all tested models, include all models been read when not specified(None)
labels:          list<str>, contains dir names in ./plot that have all_metrics.csv located
alt_labels:      list<str>, alternative name for each element in labels, should have the same length as that of labels
col_num_per_plot:      int, number of columns per figure(for cleaner figure)
experiment_name:       str, the name of dir in ./fig where all resulting figures belong
rex:                   str, regular expression for filtering wanted metrics
'''

if __name__ == '__main__':
    metric_names = ["MSE", "RMSE", "RMSPE", "MAE", "MAPE", "WMAPE", "RSE", "VWMAPE", "corr"]
    # models = None
    # models = ["Persistence", "MA"]  # baselines only
    models = ["convGRU", 'stationary_convGRU', 'znorm_convGRU',
              "transformer", "stationary_transformer", 'znorm_transformer']  # models only
    # models = ["transformer", "stationary_transformer", 'znorm_transformer']    # transformer only
    # models = ["convGRU", 'stationary_convGRU', 'znorm_convGRU'] # convGRU only
    base = "layers[{}]_EC_i168s0o168_rate24trate24_norm[std]scale[None]_bypass[None]TE[None]split[False]"
    labels = [base.format(i) for i in range(1, 9)]
    alt_labels = ["Layers-{}".format(i) for i in range(1, 9)]
    col_num_per_plot = 10
    # experiment_name = "test"
    experiment_name = "layers_ablation"
    rex = '$'

    try:
        assert len(alt_labels) == len(labels)
        alt_labels = dict(zip(labels, alt_labels))
    except Exception as ex:
        alt_labels = dict(zip(labels, labels))
    max_index_len = max([len(alt_labels[k]) for k in alt_labels])
    base_path = os.path.sep.join(["fig"]) if len(labels) == 1 else os.path.sep.join(["fig", "ablation"])
    try:
        os.mkdir(base_path)
    except:
        pass
    try:
        os.mkdir(os.path.sep.join([base_path, experiment_name]))
    except:
        pass
    for name in metric_names:
        all_months = pd.DataFrame()
        try:
            os.mkdir(os.path.sep.join([base_path, experiment_name, name]))
        except:
            pass
        for month in range(1, 13):
            all = pd.DataFrame()
            mydate = datetime.datetime.strptime(str(month), "%m")
            month_name = mydate.strftime("%B")
            month_abbr = mydate.strftime("%b")
            for label in labels:
                experiment_label = label + "_test_on_{}".format(month_abbr)
                path = "plot/{}/{}".format(experiment_label, "all_metric.csv")
                is_monthly_result_existing = os.path.exists(path)
                if not is_monthly_result_existing:
                    experiment_label = label
                    path = "plot/{}/{}".format(experiment_label, "all_metric.csv")
                try:
                    metrics = pd.read_csv(path, index_col=0)
                    metric = metrics.filter(regex='^' + name + rex, axis='columns').round(5)
                    metric = pd.concat({alt_labels[label]: metric}, names=['exp_name', 'models'])
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
                    unstacked = all.unstack().droplevel(0, axis=1)[alt_labels.values()]
                    unstacked = unstacked.reindex(models) if models is not None else unstacked
                    for i, v in enumerate(range(0, len(unstacked), col_num_per_plot)):
                        tmp = unstacked[v:v + col_num_per_plot]
                        title = "{} Test on {}".format(name, month_name)
                        fig_path = os.path.sep.join([base_path, experiment_name, name, "{}_{}".format(title, i)])
                        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                                                figsize=(2 * tmp.shape[0], 9 + .1 * tmp.shape[1]))
                        axs[1].axis('off')
                        axs[1].axis('tight')
                        ax = axs[0]
                        tmp.plot.bar(title=title, ax=ax, legend=True)
                        ax.axes.get_xaxis().set_visible(False)
                        t = tmp.transpose()
                        pd.plotting.table(ax, t, loc='bottom', rowLoc='right', colLoc='center')
                        ax.legend(loc='lower right', fontsize=5)
                        # plt.show()
                        plt.savefig(fig_path)
                        plt.clf()
                all.columns = pd.MultiIndex.from_product([all.columns, [month_name]])
                all_months = pd.concat([all_months, all], axis=1)
                if not is_monthly_result_existing:
                    break
        assert len(all_months.columns) > 0
        mean = all_months.groupby(axis=1, level=0).mean().round(3).unstack().droplevel(0, axis=1)[alt_labels.values()]
        mean = mean.reindex(models) if models is not None else mean
        std_err = all_months.groupby(axis=1, level=0).sem().round(3).unstack().droplevel(0, axis=1)[alt_labels.values()]
        std_err = std_err.reindex(models) if models is not None else std_err
        for i, v in enumerate(range(0, len(mean), col_num_per_plot)):
            mean_tmp = mean[v:v + col_num_per_plot]
            std_tmp = std_err[v:v + col_num_per_plot]
            title = ('Mean {}'.format(name) + "" if std_tmp.isna().any().any() else " with SE")
            fig_path = os.path.sep.join([base_path, experiment_name, name, "{}_{}".format(title, i)])
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                                    figsize=(2 * mean_tmp.shape[0], 9 + .1 * mean_tmp.shape[1]))
            axs[1].axis('off')
            axs[1].axis('tight')
            ax = axs[0]
            mean_tmp.plot.bar(yerr=std_tmp, capsize=3, title=title, ax=ax,
                              legend=True, layout="constrained")
            ax.axes.get_xaxis().set_visible(False)
            ax.legend(loc='lower right', fontsize=5)
            t = (mean_tmp.astype(str) + ("" if std_tmp.isna().any().any() else'±' + std_err.astype(str))).transpose()
            pd.plotting.table(ax, t, loc='bottom', rowLoc='right', colLoc='center')
            # plt.show()
            plt.savefig(fig_path)
            plt.clf()
pass
