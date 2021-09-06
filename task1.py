#-*-coding: UTF-8 -*-
# #ofhash=50 will exceed the time limit a little bit
# take number_of_hash=45, band=45 => row=1
# make sure to comment all print when running full dataset

from pyspark import SparkContext
from pyspark import SparkConf
from os import sys
import random
import json
import itertools
import time

# spark settings
def CreateSparkContext():
    sConf = SparkConf().setMaster("local")\
            .setAppName('task1')\
            .set("spark.ui.showConsoleProgress", "false")\
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=sConf)
    #print("master"+sc.master)
    sc.setLogLevel("ERROR")
    # SetPath(sc) #for local use
    # spark = SparkSession.builder.config(conf=sConf).getOrCreate()

    return sc

# Tried but not used functions

# def MinHash(n):
    # return 
# def read_file(r):
#     r = r[1:-1]
#     ratings = r.split(',')
#     # print ratings
#     ratings = ratings[1:3]
#     x = []
#     for k in ratings:
#         x.append(k.split(':')[1])
#     # print(x)
#     return ((x[0], x[1]), 1)

# def min_hash(mat, hashed):
    # ret = []
    # unum = len(mat)
    # bnum = len(mat[0])
    # for j in range(unum):
        # for i in range(bnum):
            # if mat[j][i] == 1:
                # ret.append(i)
                # break
    # return ret

# return the hash function
# use f=(a*x+b)%m
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

def get_hashed_user(x):
    # input value x to hash functions
    # put all results in a list
    hfs = []
    for func in hash_func_list:
        hfs.append(func(x))
    return hfs

def flatt(x):
    res = []
    for item in x[0]:
        tmp = tuple([item, x[1]])
        res.append(tmp)
    return res
    
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
    
    # i hopping; still need a counter while looping
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

# for convenience on finding the main part, not necessary
if __name__ == '__main__':
    
    start = time.time()
    # check if input correctly (not wasting time :)
    if len(sys.argv) != 3:
        #print "Please input the names of input file and output file"
        exit()
    #
    sc= CreateSparkContext()
    
    # read json as rdd and only keep {usr_id, bus_id}
    infile = sc.textFile(sys.argv[1])\
                .map(lambda row: json.loads(row))\
                .map(lambda x: (x['user_id'], x['business_id']))
    # print(infile.count())
    # infile.show(3)
    
    # actually not necessary in the new algorithm, but for not messing up more codes, just keep it
    ratings = infile.map(lambda x: (x, 1)) 
    # rate_dic = ratings.collectAsMap()
    # ratings = ratings.map(lambda x:(x[0], x[1])).collectAsMap()
    # print(ratings)
    
    # assign all users (distinct) idx
    user_idx = ratings.map(lambda x:x[0][0]).distinct().zipWithIndex()
    # use dict to map user to idx
    all_user = user_idx.map(lambda x: {x[0]: x[1]})
    # count the total number of users
    user_num = all_user.count() # 20000+
    
    # all business (distinct) to idx
    busi_idx = ratings.map(lambda x:x[0][1]).distinct().zipWithIndex()
    all_busi = busi_idx.map(lambda x: {x[0]: x[1]})
    # count the total number of business # not useful in this task
    # busi_num = all_busi.count()
    
    # x.items() returns dict_item(): a list of (key, value) to do the flatMap()
    # to (dict) map collections (in java?) (key -> value)
    # real dict; for easy searching idx
    busi_dict = all_busi.flatMap(lambda x: x.items()).collectAsMap()
    user_dict = all_user.flatMap(lambda x: x.items()).collectAsMap()
    # print(busi_dict, user_dict)
    
    # business_idx to [users_idx] prepare for future jaccard calculation
    buidx = infile.map(lambda x: (busi_dict[x[1]], [user_dict[x[0]]])) \
                .reduceByKey(lambda a,b: a+b)
    # bidx to uidx unique pairs
    bumat = buidx.map(lambda x: {x[0]: list(set(x[1]))})\
                .flatMap(lambda x:x.items())\
                .collectAsMap()
    
    # user_idx to [business_idx] prepare for signature matrix
    ubidx = infile.map(lambda x: (user_dict[x[0]], [busi_dict[x[1]]])) \
                .reduceByKey(lambda a,b: a+b)
    # uid to bid unique pairs
    ubmat = ubidx.map(lambda x: {x[0]: list(set(x[1]))})\
                .flatMap(lambda x:x.items()) # this needs to be list not dict for rdd join
                #.collectAsMap()
    # print("finish mat pre")
    
    # also need index to busi dict for output check
    idx_to_busi = busi_idx.map(lambda x: {x[1]: x[0]})\
                        .flatMap(lambda x: x.items())\
                        .collectAsMap()
    # print(idx_to_busi)
    # exit()
    # try hard-coded hash
    # lista = [12281, 5, 11, 61, 221, 1]
    # listb = [1, 281, 61, 1, 5, 3881]
    # lenh = len(lista)
    
    hash_num = 45
    hash_func_list = hash_func(hash_num, 2*user_num) # Choosing 2*user_num based on Piazza discussion
    hashed_user = all_user.flatMap(lambda x: x.items()) \
                .map(lambda x: (user_dict[x[0]], get_hashed_user(x[1])))
    
    # use join to calculate the signature matrix using minhash
    # join as something like (uidx, (bidx, [hashed_usrs]))
    hbmat_join = ubmat.leftOuterJoin(hashed_user)
    # remove the uid; find min value
    hbmat_rdd = hbmat_join.map(lambda x: x[1])\
                .flatMap(lambda x: flatt(x))\
                .reduceByKey(lambda a, b: get_min(a, b))
    
    # print(hbmat_join.take(10))
    # print(hbmat_rdd.take(10))
    # exit()
    
    # hbmat = [[infi]*lenh for _ in range(busi_num)]
    
    # for i in range(lenh):
    #     print("hash No."+str(i))
    #     hashed = generate_list(origin, lista[i], listb[i], user_num)
    #     for j in range(busi_num):
    #         for k in range(user_num):
    #             if bumat[j][hashed[k]] == 1:
    #                 hbmat[j][i] = k
    #                 break

    # print(middle-start, end-middle)     
    # print(hbmat)
    
    # find all candidates
    candidates_lst = hbmat_rdd.flatMap(lambda x: split_sig(x))\
                    .reduceByKey(lambda a,b: a+b)

    # only bids w/ at least two users' ratings 
    # pair together as candidates
    candidates=candidates_lst.map(lambda x: x[1])\
                    .filter(lambda x: len(x)>=2)\
                    .flatMap(lambda x: candidate_pairs(x))
    
    # jac_mat = [[0.0]*busi_num for _ in range(busi_num)]  
    # for i in range(busi_num):
    #     if i%100 == 0:
    #         print "current candidate progress:"+str(i)
    #     for j in range(i+1, busi_num):
    #         for p in range(3):
    #             if hbmat[i][p*2] == hbmat[j][p*2] and hbmat[i][p*2+1] == hbmat[j][p*2+1]:
    #                 candidates.add((i, j))
    #                 break
    
    # print candidates.take(10)
    
    # sorting makes life easier :
    candidates = candidates.sortBy(lambda x: x).collect()
    # print(candidates)
    

    #calculate Jaccard similarity and filter out
    res = []
    finished_set = set() # make sure non-repeating
    for pair in candidates:
        if pair not in finished_set:
            finished_set.add(pair)
            ulist1 = set(bumat.get(pair[0]))
            ulist2 = set(bumat.get(pair[1]))
            # intersection/union
            jacc = len(ulist1.intersection(ulist2))*1.0/len(ulist1.union(ulist2))
            if jacc >= 0.05:
                tmp = dict()
                tmp["b1"] = idx_to_busi[pair[0]]
                tmp["b2"] = idx_to_busi[pair[1]]
                tmp["sim"] = jacc
                res.append(tmp)
    
    out_path = (sys.argv[2])
    out_file = open(out_path, 'w+')
    for item in res:
        out_file.writelines(json.dumps(item) + "\n")
    out_file.close()
    end = time.time()
    print("Duration:"+str(end-start))