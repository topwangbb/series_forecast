# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 10:07
# @Author  : zhongxj
# @FileName: restful_test.py
import base64

import requests
import time
import json


payload = {"order_uuid": "AI_INDEX_001",
        "predict_n":7,
        "bound":0.03,
        "p":-1,
        "q":-1,
        "m":-1
}

# 时序预测的示例URL地址
url = 'http://localhost:2222/AH_series_forecast'


start = time.time()

# 获取接口
response = requests.post(url,data=payload,files={'file': ('data.csv', open("D:\qq文件\AI_省_4G数据流量_4734742702149169682.csv", 'rb'))})
print(response)
print(time.time()-start)

response.encoding = 'utf-8'
# 吧csv和返回信息一起在content输出为json
result = response.json()
print(result)
print(json.dumps(result, ensure_ascii=False, indent=4))

