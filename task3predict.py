from pyspark import SparkContext
from pyspark import SparkConf
from os import sys
import math
import json
import time
import itertools
from collections import defaultdict

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
    if x[0]==None or x[1][0]==None:
        return None
    key = tuple([x[0], x[1][0]])
    for item in x[1][1]:
        if x[1][0] != item[0]:
            new_dict[key][item[0]]=item[1]
    
    return new_dict

# input: {(preu, preb): {b1:r1, b2:r2,...}}
# let N = 8
def predict_item(x):
    pred_pairs = list(x.keys())
    N_lst = []
    values = list(x.values())
    try:
        for key, value in values[0].items():  
            bb_pair = [pred_pairs[0][1], key]
            search_key = tuple(sorted(bb_pair)) # order matters
            if model_bb_s.get(search_key):
            #find N neighbors
                N_lst.append([model_bb_s[search_key], value]) # (similarity, rating)
    
        N_neighbors = sorted(N_lst, reverse=True)[:8]
        # calculate rating
        # get numerator and denominator
        num = 0
        denom = 0
        for item in N_neighbors:
            num += item[0]*item[1] # weighted: sim*rating
            denom += item[0] # total weight
        
        if num == 0 or denom == 0:
            rating = 0
        else:
            rating = num/denom
    
        return (pred_pairs[0][0], pred_pairs[0][1], rating)
    except:
        #try:
        #    #print(pred_pairs)
        #    return (pred_pairs[0][0], pred_pairs[0][1], bus_avg[pred_pairs[0][1]])
        #except:
            #print(x.keys())
        return('u', 'b', 0)

def predict_user(x):
    pred_pairs = list(x.keys())
    N_lst = []
    values = list(x.values())
    try:
        for key, value in values[0].items():  
            uu_pair = [pred_pairs[0][1], key]
            search_key = tuple(sorted(uu_pair)) # order matters
            if model_uu_s.get(search_key):
            #find N neighbors
                N_lst.append([model_uu_s[search_key], value]) # (similarity, rating)
    
        N_neighbors = sorted(N_lst, reverse=True)[:24]
        # calculate rating
        # get numerator and denominator
        num = 0
        denom = 0
        for item in N_neighbors:
            num += item[0]*item[1] # weighted: sim*rating
            denom += item[0] # total weight
        
        if num == 0 or denom == 0:
            rating = 0
        else:
            rating = num/denom
    
        return (pred_pairs[0][1], pred_pairs[0][0], rating)
    except:
        #try:
        #    #print(pred_pairs)
        #    return (pred_pairs[0][0], pred_pairs[0][1], user_avg[pred_pairs[0][1]])
        #except:
            #print(x.keys())
        return('u', 'b', 0)
    
     

if __name__ == '__main__':
    start = time.time()
    # print sys.argv
    if len(sys.argv) != 6:
        print("Please input correct files")
        exit()
    sc= CreateSparkContext()
    
    #print("yoho")
    train_file = sc.textFile(sys.argv[1]).map(lambda row: json.loads(row)) \
            .map(lambda x: (x['user_id'], x['business_id'], x['stars']))
    test_file = sc.textFile(sys.argv[2]).map(lambda row: json.loads(row)) \
            .map(lambda x: (x['user_id'], x['business_id']))
    
    
    if sys.argv[5] == "item_based":
        model_file = sc.textFile(sys.argv[3]).map(lambda row: json.loads(row)) \
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
        
        #print(list(model_bb_s.items())[0])
        #exit()
    
        train_test_join = test_u_b.leftOuterJoin(train_u_br) \
                            .filter(lambda x: x[1][0] != None) \
                            .map(lambda x: to_dict(x)) \
                            .filter(lambda x: x!=None)
        
        #error_check = test_u_b.leftOuterJoin(train_u_br) \
        #                    .filter(lambda x: x[1][0] != None).collect()
        #with open('../work/textfile', 'w+') as fi:
        #    for item in error_check[:1000]:
        #        fi.writelines(str(item)+'\n')
        #fi.close()
    
        res = train_test_join.map(lambda x: predict_item(x)) \
                            .filter(lambda x: x[2]!=0) \
                            .collect()
        
        out_path = (sys.argv[4])
        out_file = open(out_path, 'w+')
        for item in res:
            out_file.writelines(json.dumps({"user_id": item[0], "business_id": item[1], "stars": item[2]}) + "\n")
        
        out_file.close()
        #print(res[:10])
        #exit()
        
    elif sys.argv[5] == "user_based":
        model_file = sc.textFile(sys.argv[3]).map(lambda row: json.loads(row)) \
            .map(lambda x: (x['u1'], x['u2'], x['sim']))
        train_b_ur = train_file.map(lambda x: (x[1], [(x[0], x[2])])) \
                            .reduceByKey(lambda a,b: a+b)
        test_b_u = test_file.map(lambda x: (x[1], x[0]))
        model_uu_s = model_file.map(lambda x: {tuple(sorted((x[0], x[1]))): x[2]}) \
                            .flatMap(lambda x: x.items()) \
                            .collectAsMap()
        user_avg = sc.textFile('../resource/asnlib/publicdata/user_avg.json') \
                    .map(lambda row: json.loads(row)).map(lambda x: dict(x)) \
                    .flatMap(lambda x: x.items()).collectAsMap()
        
        train_test_join = test_b_u.leftOuterJoin(train_b_ur) \
                            .filter(lambda x: x[1][1] != None) \
                            .map(lambda x: to_dict(x)) \
                            .filter(lambda x: x!=None)
                    
        res = train_test_join.map(lambda x: predict_user(x)) \
                            .filter(lambda x: x[2]!=0) \
                            .collect()
                
        out_path = (sys.argv[4])
        out_file = open(out_path, 'w+')
        for item in res:
            out_file.writelines(json.dumps({"user_id": item[0], "business_id": item[1], "stars": item[2]}) + "\n")
        
        out_file.close()
        
    end = time.time()
    print("Duration: "+str(end-start))