import pandas as pd
from pathlib import Path
import numpy as np
import os
from datetime import datetime

def fill_cloud(df):
    # df['ShortWaveDown'].apply(lambda x: float(x))
    # print(df.index[0])
    df2 = pd.DataFrame()
    path = 'output.lst'
    a = []
    b = []
    with open(path) as f:
        lines = f.readlines()
        for j in range(len(lines)):
            month = int(lines[j][41:43])
            day = int(lines[j][43:45])
            hour = int(lines[j][46:48])
            minute = int(lines[j][48:50])
            d = datetime(2020, month, day, hour, minute)
            c = lines[j][57:58]
            b.append(c)
            a.append(d)
            # if df.index[j] in a:
            #     print(df.index[j])
            #     df['twoclass'][j] = lines[j][48:50]
        df2 = pd.DataFrame({ 'datetime' : a, 'twoClass' : b }) 
    # df['ShortWaveDown'].apply(lambda x: float(x))
    # df['ShortWaveDown'].fillna(method='ffill', inplace=True)
    # df['ShortWaveDown'] = df['ShortWaveDown'].fillna(method='ffill', inplace=True)
    # print(df)
    # if df.isnull():
    #     df = df.join(df)
    # df3 = pd.concat([df,df2], axis=1
    print(df2)
    # df.merge(df2, left_index=True, right_index=True)
    df3 = pd.merge(df, df2, how='left', on=['datetime'])
    df3.set_index('datetime',inplace = True)
    print(df3)
    return df3

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
    output_path = Path('2020new2.csv')

    df_data.to_csv(output_path, encoding='utf8')
    # print("generateEntireDataset completed: ",output_path)


if __name__ == '__main__':
    generateEntireDataset("2020new")
