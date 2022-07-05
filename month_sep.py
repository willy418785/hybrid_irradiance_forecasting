import main 
from pyimagesearch import parameter
import tensorflow as tf
import pandas as pd
import os
from pathlib import Path
import gc
############################
# test all month
############################

if __name__ == '__main__':
    olabel=parameter.experient_label
    df_dict = {}
    # 各地區有資料的月份
    # renheo    1-12        
    # ITRISouth: 3-8    
    # nafco1 : 1-8
    # nafco2 : 1-6
    # nafco3 : 1-8
    # range 2,13 以測試月為準 2 ~ 12 這樣
    # 3:9 => train二月 test三月 val一月，最多test到八月
    for m in range(2,9):
        parameter.test_month = m      #2,3,4,5,6,7,8
        parameter.experient_label = olabel+"_"+str(m)

        metircLoger=main.run()

        # metircLoger 就是訓練結束後回傳的各metric結果 並且在這以下取用彙整
        for metircKey in parameter.csvLogMetrics:
            if df_dict.get(metircKey) is None:
                df_dict[metircKey] = pd.DataFrame({'Model': list(parameter.model_list)})
            # print(metircLoger[metircKey])
            # print(df_dict)
            # print(df_dict[metircKey][str(m)])
            df_dict[metircKey][str(m)] = metircLoger[metircKey]

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
    csvLogPath ="csvLogs/{}/".format(olabel)
    try:
        os.mkdir(Path(csvLogPath))
    except:
        pass
    for k,v in df_dict.items():
        v['AVG'] = (v.copy()).mean(numeric_only=True, axis=1)  # v append average
        v.to_csv(Path(csvLogPath+k+"_"+olabel+".csv",index=False))
    print("===========================================")
    print(olabel)