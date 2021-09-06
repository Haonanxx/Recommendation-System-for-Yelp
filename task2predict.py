from pyspark import SparkContext
from pyspark import SparkConf
from os import sys
import math
import json
import re
import string
import time


def CreateSparkContext():
    # driver and executor memory set to 4g due to the grading limit
    sConf = SparkConf().setAppName('task2_predict') \
                    .set("spark.ui.showConsoleProgress", "false") \
                    .set("spark.executor.memory", "4g") \
                    .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=sConf)
    #print "master "+sc.master
    sc.setLogLevel("ERROR")
    #SetPath(sc)
    # spark = SparkSession.builder.config(conf=sConf).getOrCreate()

    return sc

#def SetPath(sc):
#    global Path
#    if sc.master[0:5] == "local":
#        Path = "file:/home/yudi/Code/Eclipse/Workspace/Project1/"
#    else:
#        Path = "hdfs://master:9000/user/yudi"

def convertToidx(x, user_list, busi_list):
    user_id = x['user_id']
    busi_id = x['business_id']
    if (user_id not in user_list.keys()) or (busi_id not in busi_list.keys()):
        return (-1, -1)
    # print (user_list[user_id], busi_list[busi_id])
    return (user_list[user_id], busi_list[busi_id])

def cosine_similarity(x, user_profile, busi_profile):
    # print(x)
    u = set(user_profile[x[0]])
    b = set(busi_profile[x[1]])
    inter = len(u.intersection(b))
    sqrt = math.sqrt(len(u)*len(b))
    sim = inter*1.0/sqrt
    #print ((x[0], x[1]), sim)
    return ((x[0], x[1]), sim)

if __name__ == '__main__':
    start = time.time()
    if len(sys.argv) != 4:
        #print "Please input the names of input file, model file and output file"
        exit()
    sc= CreateSparkContext()
    model = sc.textFile(sys.argv[2]).map(lambda row: json.loads(row))
    #print model.collect()
    user_profile = model.filter(lambda x: x['type'] == 'u')\
                        .map(lambda x: (x['index'], x['profile']))\
                        .collectAsMap()
    idx_to_user = model.filter(lambda x: x['type'] == 'u')\
                        .map(lambda x: (x['index'], x['id']))
    user_list = idx_to_user.map(lambda x: (x[1], x[0])).collectAsMap()
    # print user_list
    idx_to_user = idx_to_user.collectAsMap()
    busi_profile  = model.filter(lambda x: x['type'] == 'b')\
                        .map(lambda x: (x['index'], x['profile']))\
                        .collectAsMap()
    idx_to_busi = model.filter(lambda x: x['type'] == 'b')\
                        .map(lambda x: (x['index'], x['id']))
    busi_list = idx_to_busi.map(lambda x: (x[1], x[0])).collectAsMap()
    # print busi_list
    idx_to_busi = idx_to_busi.collectAsMap()
    
    valid_predict = sc.textFile(sys.argv[1]).map(lambda row: json.loads(row)) \
                    .map(lambda x: convertToidx(x, user_list, busi_list))\
                    .filter(lambda x: x[0]>=0 and x[1]>=0)
            
    predict_test = valid_predict.map(lambda x: cosine_similarity(x, user_profile, busi_profile)) \
                    .filter(lambda x: x[1]>0.01)\
                    .collectAsMap()
    
    res = []
    for d in predict_test.items():
        tmp = {}
        tmp['user_id'] = idx_to_user[d[0][0]]
        tmp['business_id'] = idx_to_busi[d[0][1]]
        tmp['sim'] = d[1]
        res.append(tmp)
    
    out_path = (sys.argv[3])
    out_file = open(out_path, 'w+')
    for item in res:
        out_file.writelines(json.dumps(item) + "\n")
    out_file.close()
    end = time.time()
    print("Duration: "+str(end-start))