from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
import scorecardpy as sc
import joblib
import os
import logging
import logging.handlers
import time
#  忽略弹出的warnings
import warnings
warnings.filterwarnings('ignore')
from scorecardpy.woebin import woepoints_ply1

#日志打印设置
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = logging.handlers.TimedRotatingFileHandler('model.log', when='midnight',encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()#往屏幕上输出
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
app.config['JSON_AS_ASCII'] = False

#通用变量
modelDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+ '/model/'
#异常定义
code10001 = {'code':'10001','errorType':'KeyError','errorMsg':'输入特征错误:'}
code10002 = {'code':'10002','errorType':'ValueError','errorMsg':'输入值错误：'}
code10003 = {'code':'10003','errorType':'UncheckException','errorMsg':'未知错误:'}
code10009 = {'code':'10009','errorType':'IOException','errorMsg':'模型文件不可达:'}
#缓存变量
modelPath,binsPath='',''
bins,model=[],[]

#传参校验是否存在
def keyIsExist(data,key):
    for i in data.keys():
        if(i==key):
            return True
    return False
#模型文件路径校验
def modelFilePathCheck(request):
    if  keyIsExist(request,'modelFilePath'):
        modelFilePath = request['modelFilePath']
    else:
        modelFilePath = modelDir + 'default/lr.pkl'
    return modelFilePath
#原始数据进行woe编码
def applyBinMap(X_data, bin_map, var):
    """
    将最优分箱的结果WOE值对原始数据进行编码
    ------------------------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, map table
    ------------------------------------------------
    Return
    bin_res: pandas Series, result of bining
    """
    x = X_data[var]
    bin_map = bin_map[bin_map['var_name'] == var]
    bin_res = np.array([0] * x.shape[-1], dtype=float)

    for i in bin_map.index:
        upper = bin_map['max'][i]
        lower = bin_map['min'][i]
        if lower == upper:
            x1 = x[np.where(x >= lower)[0]]
        else:
            x1 = x[np.where((x >= lower) & (x < upper))[0]]  #会去筛选矩阵里面符合条件的值
        mask = np.in1d(x, x1)    #用于测试一个数组中的值在另一个数组中的成员资格,返回布尔型数组
        bin_res[mask] = bin_map['WOE'][i]   #将Ture的数据替换掉

    bin_res = pd.Series(bin_res, index=x.index)
    bin_res.name = x.name

    return bin_res
#判断字符串是否为数字
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
def calculateFeatures(dt, card):
    dt = dt.copy(deep=True)
    # card # if (is.list(card)) rbindlist(card)
    if isinstance(card, dict):
        card_df = pd.concat(card, ignore_index=True)
    elif isinstance(card, pd.DataFrame):
        card_df = card.copy(deep=True)
    # x variables
    xs = card_df.loc[card_df.variable != 'basepoints', 'variable'].unique()
    dat = {}

    # loop on x variables
    for feature in xs:
        cardx = card_df.loc[card_df['variable']==feature]
        dtx = dt[[feature]]
        # score transformation
        dtx_points = woepoints_ply1(dtx, cardx, feature, woe_points="points")
        dtx_points.columns = [feature]
        dat[feature]= str(dtx_points.values[0][0])
    return dat


@app.route('/model/execute',methods=['POST'])
def execute_data():
    start = time.time()
    try:
        global modelPath,binsPath,bins,model
        logger.info('入参：%s',str(request.get_data()))
        request_data = request.get_json() #获取传入数据
        paramsJson = request_data['paramsData']
        modelFilePath = modelFilePathCheck(request_data)

        if (type(paramsJson) == type({})):

            #原始数据中数值型字符串转为int64
            items = paramsJson.items();
            for key,value in items:
                if (is_number(value)):
                    paramsJson[key] = int(value)
            logger.info("paramsJson:%s",paramsJson)
            #构建dataFrame
            df = pd.json_normalize(paramsJson)

            if (modelPath != modelFilePath):
                logger.info('调用模型路径发生变化，重新加载模型！')
                logger.info('global modelFilePath:%s',modelPath)
                logger.info('param modelFilePath:%s',modelFilePath)
                modelPath = modelFilePath
                #导入模型
                model = joblib.load(modelFilePath)
            else:
                logger.info('调用模型路径未发生变化，使用缓存中模型。')
                logger.info('global modelFilePath:%s',modelPath)

            #原始数据转换为woe值
            bins = model.bins
            df_woe = sc.woebin_ply(df, bins)
            #打标签
            label = model.predict(df_woe)[0]

            #构建评分卡
            card = sc.scorecard(bins, model, df_woe.keys())
            #评分
            score = sc.scorecard_ply(df, card,only_total_score=False, print_step=0)
            #计算每个特征的得分
            featureScore = {}
            # featureScore = calculateFeatures(df, card)
            if isinstance(card, dict):
                card_df = pd.concat(card, ignore_index=True)
            elif isinstance(card, pd.DataFrame):
                card_df = card.copy(deep=True)
            # x variables
            xs = card_df.loc[card_df.variable != 'basepoints', 'variable'].unique()
            for i in xs:
                featureScore[i] = score[i + '_points'][0]

            result = {}
            result['code'] = '00000'
            result['score'] = str(score['score'][0])
            result['label'] = str(label)
            result['featureScore'] = featureScore
            end = time.time()
            logger.info("运行结果：%s,模型执行耗时：%s",result,end-start)
            return jsonify(result)

        code10002['errorMsg']='输入值错误：请传入json格式参数'
        return jsonify(code10002)

    except KeyError as e:
        logger.info(e)
        code10001['errorMsg']='输入特征错误：' + str(e)
        return jsonify(code10001)

    except ValueError as e:
        logger.info(e)
        code10002['errorMsg']='输入值错误：' + str(e)
        return jsonify(code10002)

    except Exception as e:
        logger.info(e)
        code10003['errorMsg']='未知错误：' + str(e)
        return jsonify(code10003)

@app.route('/model/features',methods=['POST'])
def features_data():
    start = time.time()
    try:
        logger.info(str(request.get_data()))
        request_data = request.get_json() #获取传入数据
        modelFilePath = modelFilePathCheck(request_data)
        if os.path.isfile(modelFilePath):
            #调用model文件
            model = joblib.load(modelFilePath)
            bins = model.bins
            result = {}
            result['code'] = '00000'
            result['features'] = list(bins.keys())
            end = time.time()
            logging.info('特征查询耗时：%s',end-start)
            return jsonify(result)

        return jsonify(code10009)

    except KeyError as e:
        logger.info(e)
        code10001['errorMsg']='输入参数错误：' + str(e)
        return jsonify(code10001)



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5003)  # 指定ip:port
    app.run(threaded=True) #开启多线程

print('运行结束')
