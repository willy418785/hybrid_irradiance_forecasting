import pandas as pd
from pathlib import Path
import numpy as np
import os
from datetime import datetime
from scipy.ndimage.interpolation import shift



def fill_cloud(df):
    # df['ShortWaveDown'].apply(lambda x: float(x))
    # print(df.index[0])
    df2 = pd.DataFrame()
    x = df['ShortWaveDown']
    # b = np.diff(x, n=1)
    array = x
    result5 = shift(array, 5, cval=0)
    result10 = shift(array, 10, cval=0)

    interval = 5
    diff5 = list()
    for i in range(interval, len(x)):
        value = x[i] - x[i - interval]
        diff5.append(value)
    # print(diff5)
    for i in range(len(df['ShortWaveDown'])-len(diff5)):
        # diff5 = np.append(diff5,0)
        diff5 = np.insert(diff5,0,0)
    
    interval = 10
    diff10 = list()
    for i in range(interval, len(x)):
        value = x[i] - x[i - interval]
        diff10.append(value)
    # print(diff10)
    for i in range(len(df['ShortWaveDown'])-len(diff10)):
        # diff10 = np.append(diff10,0)
        diff10 = np.insert(diff10,0,0)
    # print(result5)
    df.insert(df.shape[1],'difference5', diff5)
    df.insert(df.shape[1],'before5', result5)
    df.insert(df.shape[1],'difference10', diff10)
    df.insert(df.shape[1],'before10', result10)
    # df['difference'] = b
    print(df)
    return df

def generateEntireDataset(region):
    df_data = pd.read_csv(Path(region + '.csv'), keep_date_col=True,
                           parse_dates=['datetime'],
                           index_col="datetime")  # 如果原本的csv有header，使用header=None時需要 , skiprows=1，因為她會把header擠到第一row
    # df_data = df_data.sort_index()
    # (option) concat history weather 加入歷史氣象
    print(df_data)
    # df_data=replaceNone(df_data,region)
    df_data=fill_cloud(df_data)
    # (option) concat forcast weather 加入預測氣象
    # df_data=joinForcastWeather(df_data,region)


    ## final fillna
    # df_data = df_data.fillna(method='ffill', inplace=True)
    # df_data = df_data[np.isin(df_data.index.year, paramter.process_year_list)]
    # df_data=df_data.dropna() ##todo if ffill can't fill ,like forecastweather data, then drop
    ## write
    # output_path = Path('../../preprocessedEntireDataset/dataset_WithForecastWeather_{}_{}.csv'.format(region,str(paramter.process_year_list)))
    # print(df_data)
    output_path = Path('test3.csv')

    df_data.to_csv(output_path, encoding='utf8')
    # print("generateEntireDataset completed: ",output_path)


if __name__ == '__main__':
    generateEntireDataset("test2")
