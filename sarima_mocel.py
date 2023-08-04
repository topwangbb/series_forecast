import warnings
import pandas as pd
global net, logger_info
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from datetime import timedelta
warnings.filterwarnings('ignore')
import numpy as np

def func(x):
    return round(x,2)

def version_sarima_with_pmdarima(order_uuid, df, predict_n=5, bound=0.2, defalut=False):
    """
    Auto-SARIMA（pmdarima）,模型预测自动删除初始两个单位的值
    """

    df2 = pd.DataFrame()
    for index in df['category'].unique():
        ts = df[df['category'] == index].reset_index().sort_values('date',ascending=True)
        # # 当前完整日期+未来预测的日期
        pre_date = pd.date_range(ts['date'].max(),ts['date'].max() + timedelta(days=predict_n))
        ts = ts['values']


        if defalut[2] != -1:
            m = defalut[2]
        else:
            m=7


        model = pm.auto_arima(ts, seasonal=True,stepwise=False,start_q=1,start_p=1
                              , m=m
                              ,n_jobs=-1,disp=-1,suppress_warnings=True,verbose=False
                              )
        # 替换客户自定义参数
        arima_order = list(model.order)
        if defalut[0] != -1:
            arima_order[0] = defalut[0]
        if defalut[1] != -1:
            arima_order[2] = defalut[1]

        #重新建模
        model = SARIMAX(ts, order=tuple(arima_order), seasonal_order=model.seasonal_order
                        ).fit()
        # 预测结果
        pred = model.get_prediction(end=ts.shape[0] + predict_n - 1).predicted_mean
        # 模型预测
        fcst = pred[ts.shape[0]:]


        # 输出csv
        df1 = pd.DataFrame()
        date_list = [d.strftime('%Y%m%d0000') for d in pre_date[-predict_n:].tolist()]
        df1['date'] = date_list
        df1['category'] = index
        df1['values'] = np.vectorize(func)(fcst.values)
        df1['up_values'] = np.vectorize(func)(fcst.values * (1 + bound))
        df1['down_values'] = np.vectorize(func)(fcst.values * (1 - bound))
        df2 = pd.concat([df2, df1], axis=0)

    #输出csv
    df2['date'] = df2['date'].apply(lambda x:int(x))
    return df2.reset_index(drop=True).to_json(orient='records')

