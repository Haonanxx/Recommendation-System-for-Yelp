from pyspark import SparkContext
from pyspark import SparkConf
from os import sys
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

def get_hash(a, b, m):
    def my_hash(x):
        return (a*x+b)%m
    return my_hash

# return all hash functions in a list
def hash_func(hash_num, m, seed=0):
    all_funcs = []
    
    # generate random 'a's and 'b's for random hash functions
    # initialize random
    random.seed(seed)
    # alist=[]
    # blist=[]
    # for i in range(hash_num):
    #     alist.append(random.randint(0,100000))
    #     blist.append(random.randint(0,100000))
    # no looping and append necessary
    # ref: https://www.tutorialspoint.com/generating-random-number-list-in-python
    alist = random.sample(range(0, 100000), hash_num)
    blist = random.sample(range(0, 100000), hash_num)
    
    for a, b in zip(alist, blist):
        # append a hash function to list
        f = get_hash(a, b, m)
        all_funcs.append(f)
    return all_funcs

def get_hashed_busi(x):
    # input value x to hash functions
    # put all results in a list
    hfs = []
    for func in hash_func_list:
        hfs.append(func(x))
    return hfs

def get_min(lst1, lst2):
    # return a list of min values
    res=[]
    for h1, h2 in zip(lst1, lst2):
        res.append(min(h1, h2))
    return res

# seperate hash list of each business with band=hash_num in this case
# for min-hash
def seperate_hash(busi):
    res = []
    band_num = hash_num
    iter_step = len(busi) // band_num #floor
    
    # need a counter while looping
    # ref: https://www.programiz.com/python-programming/methods/built-in/enumerate
    for i, c in enumerate(range(0, len(busi), iter_step)):
        res.append((i, tuple(busi[c: c+iter_step])))
    # return tuple to fit keys
    return tuple(res)

def split_sig(x):
    sep_hash = seperate_hash(x[1])
    res = []
    for item in sep_hash:
        tmp = tuple([item, [x[0]]])
        res.append(tmp)
    return res

def candidate_pairs(x):
    comb = list(itertools.combinations(x,2))
    return comb

if __name__ == '__main__':
    start = time.time()
    # print sys.argv
    if len(sys.argv) != 4:
        #print("Please input the names of input file, output model and stopwords file")
        exit()
    sc= CreateSparkContext()
    
    infile = sc.textFile(sys.argv[1]).map(lambda row: json.loads(row))\
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

    if sys.argv[3] == "item_based":
        #print('iiiiiii')
        
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
        out_path = (sys.argv[2])
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
        
    elif sys.argv[3] == "user_based":
        #print("Not implemented!")
        # for future searching
        uid_bidrat = infile.map(lambda x: (user_dict[x[0]], [(busi_dict[x[1]], x[2])])) \
                        .reduceByKey(lambda a,b: a+b) \
                        .filter(lambda x: len(x[1])>=11) \
                        .map(lambda x: to_dict(x)) \
                        .flatMap(lambda x: x.items()) \
                        .collectAsMap()
        # apply algorithms to reduce number of pairs, opposite use in uid and bid as task1
        # business_idx to [users_idx] prepare for signature matrix
        buidx = infile.map(lambda x: (busi_dict[x[1]], [user_dict[x[0]]])) \
                    .reduceByKey(lambda a,b: a+b) 
        # bidx to uidx unique pairs
        bumat = buidx.map(lambda x: {x[0]: list(set(x[1]))})\
                    .flatMap(lambda x:x.items()) # this needs to be list not dict for rdd join
                    #.collectAsMap()
    
        # user_idx to [business_idx] prepare for future jaccard calculation
        ubidx = infile.map(lambda x: (user_dict[x[0]], [busi_dict[x[1]]])) \
                    .reduceByKey(lambda a,b: a+b) \
                    .filter(lambda x: len(x[1])>=11)
        # uid to bid unique pairs
        ubmat = ubidx.map(lambda x: {x[0]: list(set(x[1]))})\
                    .flatMap(lambda x:x.items()) \
                    .collectAsMap()
        
        # raised a little bit from task1 to increase recall.
        hash_num = 56
        hash_func_list = hash_func(hash_num, 2*busi_num) # Choosing 2*user_num based on Piazza discussion
        hashed_busi = all_busi.flatMap(lambda x: x.items())\
                        .map(lambda x: (busi_dict[x[0]], get_hashed_busi(x[1])))
        
        # use join to calculate the signature matrix using minhash
        # join as something like (uidx, (bidx, [hashed_usrs]))
        humat_join = bumat.leftOuterJoin(hashed_busi)
        # remove the uid; find min value
        humat_rdd = humat_join.map(lambda x: x[1])\
                    .flatMap(lambda x: [(uid, x[1]) for uid in x[0]])\
                    .reduceByKey(lambda a, b: get_min(a, b))
        
        # find all candidates
        candidates_lst = humat_rdd.flatMap(lambda x: split_sig(x))\
                    .reduceByKey(lambda a,b: a+b)
        
        # only bids w/ at least three users' ratings 
        # pair together as candidates
        candidates=candidates_lst.map(lambda x: x[1])\
                    .filter(lambda x: len(x) >= 11)\
                    .flatMap(lambda x: candidate_pairs(x))
        candidates = candidates.sortBy(lambda x: x).collect()
        
        #print(candidates[:10])
        #calculate Jaccard similarity and pearson similarity
        res = []
        finished_set = set() # make sure non-repeating
        out_path = (sys.argv[2])
        out_file = open(out_path, 'w+')
        for pair in candidates:
            try:
                if pair not in finished_set:
                    finished_set.add(pair)
                    blist1 = set(ubmat.get(pair[0]))
                    blist2 = set(ubmat.get(pair[1]))
                    # intersection/union
                    jacc = len(blist1.intersection(blist2))*1.0/len(blist1.union(blist2))
                    if jacc>=0.02:
                        #compute pearson
                        u1 = idx_to_user[pair[0]]
                        u2 = idx_to_user[pair[1]]
                        bus_rat1 = uid_bidrat[pair[0]]
                        bus_rat2 = uid_bidrat[pair[1]]
                        # compute pearson similarity
                        # filter: co-rated >= 3
                        if len(set(bus_rat1.keys()).intersection(set(bus_rat2.keys()))) >= 3:
                            ps = pearson_sim(bus_rat1, bus_rat2)
                            if ps > 0:
                                out_file.writelines(json.dumps({"u1": u1, "u2": u2, "sim": ps}) + "\n")
            except:
                continue
                
        out_file.close()
                                                                  
    end = time.time()
    print("Duration: "+str(end-start))