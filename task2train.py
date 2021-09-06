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
    sConf = SparkConf().setAppName('task2_train') \
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

def process_text(content, stopwords):
    # remove ""
    # content = r
    #print r
    #print "Puncuations and Numbers"
    content = str(content.encode('utf-8')).lower()
    #res = []
    # The first 2 must have equal length
    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    #c = content.translate(str.maketrans(string.digits + string.punctuation, ' '*42))#' '*42
    # print c
    #words = re.split(r"[~\s\r\n]+", c)
    # print(words)
    # ref: https://stackoverflow.com/questions/18429143/strip-punctuation-with-regex-python
    content = re.sub(r'[^\w\s\r\n]+', ' ', content)
    # ref: https://stackoverflow.com/questions/12851791/removing-numbers-from-string
    content = ''.join(char for char in content if not char.isdigit())
    words = content.split() # list of words
    res=[]
    for w in words:
        if w not in stopwords and w != None:
            res.append(w)
    # print res
    return res

def no_dup_flat(x):
    # assign whatever, reduceByKey?
    res = []
    for word in x[1]:
        res.append(tuple([word,1]))
    return res

# counter and filter
def freq_words(contents, threshold):
    counter = dict()
    res = []
    for words in contents:
        total = len(words)
        for w in words:
            # print(w)
            if w in counter.keys():
                counter[w] += 1
            else:
                counter[w] = 1

    most_freq = max(counter.values())
    freq_count = dict()
    for c in counter.items():
        if c[1]>threshold:
            freq_count[c[0]] = c[1]
    freq_count = sorted(freq_count.items(), key=lambda x: x[1], reverse=True)
    res = []
    for fc in freq_count:
        res.append((fc[0], fc[1], total))
    return res

def calc_tf(x):
    res = []
    for fc in x[1]:
        tmp_key = tuple([x[0], fc[0]])
        tmp_val = fc[1]*1.0/fc[2]
        res.append(tuple([tmp_key, tmp_val]))
    return res

def calc_idf(x):
    res = []
    for b in x[1]:
        tmp_key = tuple([b, x[0]])
        tmp_val = math.log(file_num/len(x[1]),2)
        res.append(tuple([tmp_key, tmp_val]))
    return res
    

def flat_in_doc(d):
    return [(w, d[0]) for w in d[1]]
    

def create_combiner(v):
    t = []
    t.append(v)
    return (t, 1)

def merge(x, v):
    # x==>(list, count)
    t = []
    if x[0] is not None:
        t = x[0]
    t.append(v)
    return (t, x[1] + 1)

def merge_combine(x, y):
    t1 = []
    t2 = []
    if x[0] is not None:
        t1 = x[0]
    if y[0] is not None:
        t2 = y[0]
    t1.extend(t2)
    return (t1, x[1] + y[1])

def flat_map_2(line):
    global file_num
    rst = []
    idf_value =  file_num* 1.0/line[1][-1] 
    #print idf_value
    for doc_pair in line[1][:-1]:
        #print(doc_pair)
        for p in doc_pair:
            #print p
            #rst[(p[0], line[0])] = (p[1], idf_value, p[1] * idf_value)
            rst.append((p[0], [(line[0], p[1] * idf_value)]))
    return rst

def top200(busi):
    return (busi[0], sorted(busi[1], key=lambda x:x[1], reverse=True)[:200])

def make_busi(x):
    #(x[0], [idxs])
    val_lst = []
    for word in x[1]:
        val_lst.append(word_to_index[word[0]])

    return (x[0], val_lst)
                
        
if __name__ == '__main__':
    start = time.time()
    # print sys.argv
    if len(sys.argv) != 4:
        #print "Please input the names of input file, output model and stopwords file"
        exit()
    sc= CreateSparkContext()
    infile = sc.textFile(sys.argv[1]).map(lambda row: json.loads(row))\
                        .map(lambda x: (x['user_id'], x['business_id'], x['text']))
    # print infile.collect()
    # infile.show(3)
    
    with open(sys.argv[3]) as f:
        stopwords = []
        for word in f:
            tmp = word.rstrip()
            stopwords.append(tmp)
    # print('stop words')
    # print(stopwords)
    
    # assign all users (distinct) idx
    user_idx = infile.map(lambda x:x[0]).distinct().zipWithIndex()
    # use dict to map user to idx
    all_user = user_idx.map(lambda x: {x[0]: x[1]})
    # count the total number of users
    user_num = all_user.count() # 20000+
    
    # all business (distinct) to idx
    busi_idx = infile.map(lambda x:x[1]).distinct().zipWithIndex()
    all_busi = busi_idx.map(lambda x: {x[0]: x[1]})
    # count the total number of business 
    busi_num = all_busi.count() # 10000+ # quick check profile should have 30000+ lines
    
    # x.items() returns dict_item(): a list of (key, value) to do the flatMap()
    # to real dict; for easy searching
    busi_list = all_busi.flatMap(lambda x: x.items()).collectAsMap()
    user_list = all_user.flatMap(lambda x: x.items()).collectAsMap()
    
    # prepare for check by index when output
    idx_to_busi = busi_idx.map(lambda x: {x[1]: x[0]})\
                        .flatMap(lambda x: x.items())\
                        .collectAsMap()
    idx_to_user = user_idx.map(lambda x: {x[1]: x[0]})\
                        .flatMap(lambda x: x.items())\
                        .collectAsMap()

    # print('user profile')
    # print(user_profile)
    
    # remove stopwords and puncs and unecessary chars
    # return a list, capable for flatMap
    busi_word = infile.map(lambda x: (busi_list[x[1]], process_text(x[2], stopwords)))

    # print('busi word')
    # print(busi_word.collect()[0])
    # exit()
    
    
    # why out of memory??  too many words?
    #word_to_index = busi_word.flatMap(lambda x: x[1])\
    #                    .zipWithIndex()\
    #                    .map(lambda x: {x[0]: x[1]})\
    #                    .flatMap(lambda x: x.items()).collectAsMap()
    
    # remove duplicate words than make dict! 
    words_no_dup = busi_word.flatMap(lambda x: no_dup_flat(x))\
                        .reduceByKey(lambda a,b: a+b)
        
    word_to_index = words_no_dup.map(lambda x: x[0])\
                        .zipWithIndex()\
                        .map(lambda x: {x[0]: x[1]})\
                        .flatMap(lambda x: x.items()).collectAsMap()
    # print("word_to_index")
    # print(word_to_index.take(10))

    # deleting rare words and compute tf
    total_words = len(word_to_index)
    # print total_words
    # exit()
    threshold = total_words*1e-6
    # print threshold
    # exit()
    busi_word_tf = busi_word.map(lambda x: (x[0], [x[1]])) \
                        .reduceByKey(lambda a,b: a+b) 
        
    busi_word_tf = busi_word_tf.map(lambda x: (x[0], freq_words(x[1], threshold)))\
                        .flatMap(lambda x: calc_tf(x))
    # print 'tf'  
    
    file_num = all_busi.count()
    # print file_num

    # calculate idf
    busi_word_idf = busi_word_tf.map(lambda x: (x[0][1], [x[0][0]])) \
                        .reduceByKey(lambda a,b: a+b)
        
    busi_word_idf = busi_word_idf.map(lambda x: (x[0], list(set(x[1])))) \
                        .flatMap(lambda x: calc_idf(x))
    
    #print(busi_word_idf.take(10))
    #exit()
    
    # method1: use rdd join
    tf_idf_join = busi_word_tf.leftOuterJoin(busi_word_idf)
    # print(tf_idf_join.take(30))
    tf_idf = tf_idf_join.map(lambda x: (x[0], x[1][0]*x[1][1])) \
                    .map(lambda x: (x[0][0], [(x[0][1], x[1])])) 
    
    # method2: combineByKey with self-written functions # actually faster but recall is bad
    #tf_idf = busi_word_tf.map(lambda x: (x[0][1], (x[0][0], x[1])))\
    #            .combineByKey(create_combiner,
    #               merge,
    #               merge_combine)\
    #            .flatMap(flat_map_2)
    # print 'tf_idf'
    # print tf_idf.collect()[0]
    tf_idf_res = tf_idf.reduceByKey(lambda a,b:a+b)
    # print 'tf_idf_res'
    # print tf_idf_res[0]

    topwords = tf_idf_res.map(top200)

    busi_profile = topwords.map(lambda x: make_busi(x))\
                        .collectAsMap()    
    
    user_profile = infile.map(lambda x: (user_list[x[0]],[busi_list[x[1]]]))\
                        .reduceByKey(lambda a, b: a+b)\
                        .map(lambda x: (x[0], list(set(x[1])))) \
                        .map(lambda x: (x[0], [busi_profile[b] for b in x[1]])) \
                        .flatMap(lambda x: [(x[0], w) for w in x[1]]) \
                        .collectAsMap()
    
    # print "busi profile"
    # print busi_profile[0]

    model = []
    
    for u in user_profile.items():
        tmp = {}
        tmp['type'] = 'u'
        tmp['index'] = u[0]
        tmp['id'] = idx_to_user[u[0]]
        tmp['profile'] = u[1]
        model.append(tmp)
    
    for b in busi_profile.items():
        tmp = {}
        tmp['type'] = 'b'
        tmp['index'] = b[0]
        tmp['profile'] = b[1]
        tmp['id'] = idx_to_busi[b[0]]
        model.append(tmp)
    
    out_path = (sys.argv[2])
    out_file = open(out_path, 'w+')
    for item in model:
        out_file.writelines(json.dumps(item) + "\n")
    out_file.close()
    end = time.time()
    print("Duration:"+str(end-start))    