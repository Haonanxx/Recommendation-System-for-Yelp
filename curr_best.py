import os
from pyspark import SparkContext
from pyspark import SparkConf
import math
import json
import time
import itertools
from collections import defaultdict
import sys

def CreateSparkContext():
    # driver and executor memory set to 4g due to the grading limit
    sConf = SparkConf().setAppName('task3_predict') \
                .set("spark.ui.showConsoleProgress", "false") \
                .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=sConf)
    #print "master "+sc.master
    sc.setLogLevel("ERROR")
    #SetPath(sc)
    # spark = SparkSession.builder.config(conf=sConf).getOrCreate()

    return sc

def to_dict(x):

    # x[1][0] can't be same as item[0]
    new_dict = defaultdict(dict)
    #if x[0]==None or x[1][0]==None:
    #    return None
    key = tuple([x[0], x[1][0]])
    if x[1][1]:
        for item in x[1][1]:
            if x[1][0] != item[0]:
                new_dict[key][item[0]]=item[1]
    else:
        new_dict[key]={'-1': -1}

    if not new_dict:
        new_dict[key]={'-1': -1}

    return new_dict

# input: {(preu, preb): {b1:r1, b2:r2,...}}
# let N = 8
def predict_item(x):
    pred_pairs = list(x.keys())
    N_lst = []
    values = list(x.values())
    try:
        for key, value in values[0].items():
            if value == -1:
                raise ValueError('cold start problem')
            bb_pair = [pred_pairs[0][1], key]
            search_key = tuple(sorted(bb_pair)) # order matters
            if model_bb_s.get(search_key):
            #find N neighbors
                N_lst.append([model_bb_s[search_key], value]) # (similarity, rating)
            else:
                try:
                    N_lst.append([0.7, (0.35*usr_avg[pred_pairs[0][0]]+0.65*bus_avg[pred_pairs[0][1]])])
                except:
                    N_lst.append([0.5, bus_avg[pred_pairs[0][1]]])

        if not N_lst:
            raise ValueError('cold start problem')
        l=len(N_lst)
        if l>=25:
            N_neighbors = sorted(N_lst, reverse=True)[:25]
        else:
            N_neighbors = sorted(N_lst, reverse=True)

        # calculate rating
        # get numerator and denominator
        num = 0
        denom = 0
        for item in N_neighbors:
            num += item[0]*item[1] # weighted: sim*rating
            denom += item[0] # total weight

        if num == 0:
            rating = 0
        else:
            rating = num/denom

        return (pred_pairs[0][0], pred_pairs[0][1], rating)
    except:
        try:
        #    #print(pred_pairs)
            return (pred_pairs[0][0], pred_pairs[0][1], (0.35*usr_avg[pred_pairs[0][0]]+0.65*bus_avg[pred_pairs[0][1]]))
        except:
            #print(pred_pairs)
            return(pred_pairs[0][0], pred_pairs[0][1], 3.09)

if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    start = time.time()
    # local run
    #train_file = 'data/train_review.json'
    #model_file = 'train.model'
    # on v
    train_file = '../resource/asnlib/publicdata/train_review.json'
    model_file = 'train.model'
    test_file = sys.argv[1]
    output_file = sys.argv[2]
    sc= CreateSparkContext()

    train_file = sc.textFile(train_file).map(lambda row: json.loads(row)) \
            .map(lambda x: (x['user_id'], x['business_id'], x['stars']))
    test_file = sc.textFile(test_file).map(lambda row: json.loads(row)) \
            .map(lambda x: (x['user_id'], x['business_id']))

    model_file = sc.textFile(model_file).map(lambda row: json.loads(row)) \
        .map(lambda x: (x['b1'], x['b2'], x['sim']))
    train_u_br = train_file.map(lambda x: (x[0], [(x[1], x[2])])) \
                        .reduceByKey(lambda a,b: a+b)
    test_u_b = test_file.map(lambda x: (x[0], x[1]))
    model_bb_s = model_file.map(lambda x: {tuple(sorted((x[0], x[1]))): x[2]}) \
                        .flatMap(lambda x: x.items()) \
                        .collectAsMap()
    bus_avg = sc.textFile('../resource/asnlib/publicdata/business_avg.json') \
                .map(lambda row: json.loads(row)).map(lambda x: dict(x)) \
                .flatMap(lambda x: x.items()).collectAsMap()

    usr_avg = sc.textFile('../resource/asnlib/publicdata/user_avg.json') \
                .map(lambda row: json.loads(row)).map(lambda x: dict(x)) \
                .flatMap(lambda x: x.items()).collectAsMap()
    #print(list(model_bb_s.items())[0])
    #exit()

    train_test_join = test_u_b.leftOuterJoin(train_u_br) \
                        .map(lambda x: to_dict(x))

    #error_check = test_u_b.leftOuterJoin(train_u_br) \
    #                    .map(lambda x: to_dict(x)) \
    #                    .collect()
    #with open('error_check', 'w+') as fi:
    #    for item in error_check:
    #        fi.writelines(str(item)+'\n')
    #exit()
    #fi.close()

    res = train_test_join.map(lambda x: predict_item(x)) \
                        .collect()

    out_path = output_file
    out_file = open(out_path, 'w+')
    for item in res:
        out_file.writelines(json.dumps({"user_id": item[0], "business_id": item[1], "stars": item[2]}) + "\n")

    out_file.close()
    end = time.time()
    print("Duration: "+str(end-start))
