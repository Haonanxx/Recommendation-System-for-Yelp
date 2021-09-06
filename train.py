import os
from pyspark import SparkContext
from pyspark import SparkConf
import math
import json
import time
import itertools
from collections import defaultdict
import random

def CreateSparkContext():
    # driver and executor memory set to 4g due to the grading limit
    sConf = SparkConf().setAppName('task3_train') \
                .set("spark.ui.showConsoleProgress", "false") \
                .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=sConf)
    #print "master "+sc.master
    sc.setLogLevel("ERROR")
    #SetPath(sc)
    # spark = SparkSession.builder.config(conf=sConf).getOrCreate()

    return sc

# input: (bid, [(u1, r1), (u2, r2),...])
# want: ({bid: {u1: r1}},{bid: {u2: r2}},...}
def to_dict(x):
    #ref: https://stackoverflow.com/questions/22345951/python-idiom-for-creating-dict-of-dict-of-list
    new_dict = defaultdict(dict)
    for item in x[1]:
        new_dict[x[0]][item[0]]=item[1]
    #print(list(new_dict.items())[:3])
    #exit()
    return new_dict

# input: 1: [{u: r}, {u: r} ...] 2:[{u: r}, {u: r} ...]
# want compute ps for r in the lst
def pearson_sim(u_r1, u_r2):
    # find common users
    u1 = set(u_r1.keys()) # good for intersection computation
    u2 = set(u_r2.keys())
    common_user = list(u1.intersection(u2))
    # comput avg_r
    r1 = []
    r2 = []
    for u in common_user:
        r1.append(u_r1[u])
        r2.append(u_r2[u])
    avg1 = sum(r1)/len(r1)
    avg2 = sum(r2)/len(r2)
    # prepare numerator and denom
    num = 0.0
    denom1, denom2 = 0.0, 0.0
    for a, b in zip(r1, r2):
        num += (a-avg1)*(b-avg2)
        denom1 += (a-avg1)**2
        denom2 += (b-avg2)**2
    denom = math.sqrt(denom1*denom2)

    if denom == 0 or num == 0:
        return 0
    elif num < 0:
        return 0

    return num/denom


if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

    start = time.time()
    train_file = '../resource/asnlib/publicdata/train_review.json'
    model_file = '../train.model'
    sc= CreateSparkContext()

    infile = sc.textFile(train_file).map(lambda row: json.loads(row))\
                        .map(lambda x: (x['user_id'], x['business_id'], x['stars']))

    # assign all users (distinct) idx
    user_idx = infile.map(lambda x:x[0]).distinct().zipWithIndex()
    # use dict to map user to idx
    # x.items() returns dict_item(): a list of (key, value) to do the flatMap()
    all_user = user_idx.map(lambda x: {x[0]: x[1]})
    # count the total number of users
    #user_num = all_user.count() # 20000+

    # all business (distinct) to idx
    busi_idx = infile.map(lambda x:x[1]).distinct().zipWithIndex()
    all_busi = busi_idx.map(lambda x: {x[0]: x[1]})
    # count the total number of business
    busi_num = all_busi.count() # 10000+

    # to dict for easy searching
    busi_dict = all_busi.flatMap(lambda x: x.items()).collectAsMap()
    user_dict = all_user.flatMap(lambda x: x.items()).collectAsMap()

    # prepare for check by index when output
    idx_to_busi = busi_idx.map(lambda x: {x[1]: x[0]})\
                        .flatMap(lambda x: x.items())\
                        .collectAsMap()
    idx_to_user = user_idx.map(lambda x: {x[1]: x[0]})\
                        .flatMap(lambda x: x.items())\
                        .collectAsMap()

    # convert to (bidx, (uidx, ratings)) pairs
    bid_uidrat_rdd = infile.map(lambda x: (busi_dict[x[1]], [(user_dict[x[0]], x[2])])) \
                    .reduceByKey(lambda a,b: a+b) \
                    .filter(lambda x: len(x[1])>=3) \
                    .map(lambda x: to_dict(x)) \
                    .flatMap(lambda x: x.items())
    bid_uidrat = bid_uidrat_rdd.collectAsMap()
    #print(bid_uidrat.take(2))
    #exit()

    # get filtered bid list, prepare for combinations
    busi_lst = bid_uidrat_rdd.map(lambda x: x[0]).collect()
    #print(busi_lst[:10])
    #exit()

    # compute and output at same time to save memory
    out_path = model_file
    out_file = open(out_path, 'w+')
    for pair in list(itertools.combinations(busi_lst, 2)):
        b1 = idx_to_busi[pair[0]]
        b2 = idx_to_busi[pair[1]]
        usr_rat1 = bid_uidrat[pair[0]]
        usr_rat2 = bid_uidrat[pair[1]]
        # compute pearson similarity
        # filter: co-rated >= 3
        if len(set(usr_rat1.keys()).intersection(set(usr_rat2.keys()))) >= 3:
            ps = pearson_sim(usr_rat1, usr_rat2)
            if ps > 0:
                out_file.writelines(json.dumps({"b1": b1, "b2": b2, "sim": ps}) + "\n")
        #exit()
        #pass
    out_file.close()
