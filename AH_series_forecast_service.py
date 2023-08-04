from datetime import datetime
from flask import request, Flask
import traceback
import logging
import html
from utils import logger
from utils.ErrorCode import *
from utils.get_config import *
from model.sarima_mocel import version_sarima_with_pmdarima
global net, logger_info
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
app = Flask(__name__)
import json


#在视图函数执行后执行
@app.after_request
def apply_caching(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers['Content-Type'] = 'application/json'
    return response




@app.route('/AH_series_forecast', methods=['POST'])
def tower_recognize():
    result = []
    err_msg = None
    code = 0

    if not request.form:
        logger_info.error("Missing input paramas! ")
        raise MissingInputField()


    # 检测参数是否都在form中，文件是否在file中
    if "order_uuid" not in request.form \
            or "predict_n" not in request.form \
            or "bound" not in request.form \
            or "p" not in request.form \
            or "q" not in request.form \
            or "m" not in request.form:
        logger_info.error("Missing input paramas! ")
        raise MissingInputField()

    elif request.files:
        if not request.files['file']:
            logger_info.error("Missing input data! ")
            raise MissingInputField()

    # 检查参数是否格式正确，并进行转换

    try:
        order_uuid = html.escape(str(request.form["order_uuid"]), quote=True)
        predict_n = int(request.form["predict_n"])
        bound = float(request.form["bound"])
        p = int(request.form["p"])
        q = int(request.form["q"])
        m = int(request.form["m"])

        # 检测参数是否在合理范围内
        if (p < -1 or p > 10) or (q < -1 or q > 10) \
                    or (m<-1 or (m>-1 and m<=0)) or (bound < 0 and bound > 1):
            logger_info.error("Wrong pqm or bound format! ")
            raise WrongparamsType()
    except:
        logger_info.error("Wrong params format! ")
        raise WrongparamsType()


    # 对file中的csv转为dataframe
    try:
        file = request.files.get('file')
        df = pd.read_csv(file,encoding='UTF-8')
    except:
        logger_info.error("Wrong csv format! ")
        raise WrongDataType()

    # 检测csv数据是否满足要求
    # date用str

    try:
        df['date'] = df['date'].astype(str)
        df['category'] = df['category'].astype(str)
        df['values'] = df['values'].astype(float)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d0000')
    except:
        logger_info.error("Wrong csv data! ")
        raise WrongDataType()


    #检测缺失值不能超过1天
    df_arima = pd.DataFrame()
    #字符串转换成int
    for index in df['category'].unique():
        ts = df[df['category'] == index].reset_index()

        #预测长度大于数据的十分之一则报错
        if len(ts)/10 <=predict_n:
            logger_info.error(f"{index}data <predict_n! ")
            raise WrongparamsType()

        # 当前完整日期+未来预测的日期
        pre_date = pd.date_range(ts['date'].min(), ts['date'].max() )
        # 创建一个完整日期的dataframe
        ts1 = pd.DataFrame(columns=['date'], data=pre_date)
        ts1 = ts1.merge(ts, on='date', how='left')

        # 缺失日期检测,
        null_df = ts1[ts1['values'].isnull()]['date']
        # 判断连续间隔是否大于2天 填1，大于两天就报错
        if 1. in (null_df.diff(1).dt.days.values):
            logger_info.error(f"{index}Wrong data loss! ")
            raise WrongDataloss()
        else:
            df1 = pd.DataFrame()
            df1['date'] = pre_date
            df1['category'] = index
            df1['values'] = ts1[['values']].interpolate(method='linear', limit_direction='both')
        df_arima = pd.concat([df_arima, df1], axis=0)

    # 能力推理部分
    try:
        starttime = datetime.now()
        #调用模型
        df = version_sarima_with_pmdarima(order_uuid=order_uuid,df=df_arima,predict_n=predict_n,bound=bound
                                              ,defalut=[p,q,m])
        endtime = datetime.now()
        code = 200
        logger_info.info("spend %f seconds this time with result %s." %
                         ((endtime - starttime).total_seconds(), result))
        print(df)
    except Exception as e:
        #把报错的完整内容保存到日志文件中
        traceback.print_exc()
        code = -1
        err_msg = str(e)
        #输出ERROR:err_msg
        logger_info.error(err_msg)

    #无论try语句是否抛出异常，finally中的语句一定会被执行
    finally:
        try:
            if err_msg is None:
                result = {
                    "code": code,
                    "order_uuid": order_uuid,
                    "msg":"Successful",
                    "data": df
                }
            else:
                result = {
                    "code": code,
                    "order_uuid": order_uuid,
                    "msg": err_msg
                }
        except:
            pass

    return json.dumps(result)


if __name__ == "__main__":
    # 读取能力基本配置
    CONFIG = get_config('configuration')


    # 输出json日志形式
    logger_info = logger.JsonLogger(datetime.now().strftime('_%Y%m%d%H%M%S')).getLogger()
    # 设置日志级别
    logger_info.setLevel(logging.DEBUG)

    # 启动flask服务
    app.run(host=CONFIG["host"], port=2222)