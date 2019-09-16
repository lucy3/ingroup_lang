# get word counts in each subreddit 
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json
from collections import Counter
import operator
import re, string

ROOT = '/data0/lucy/ingroup_lang/'
WORD_COUNT_DIR = ROOT + 'logs/word_counts/'
PMI_DIR = ROOT + 'logs/pmi/'
SR_DATA_DIR = ROOT + 'subreddits/'
LOG_DIR = ROOT + 'logs/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

regex = re.compile('[%s]' % re.escape(string.punctuation))

def count_words(): 
    """
    Get word counts in each subreddit
    
    Currently askreddit is broken
    """
    log_file = open(LOG_DIR + 'counting_log.temp', 'w')
    for filename in os.listdir(SR_DATA_DIR): 
        if os.path.isdir(SR_DATA_DIR + filename): 
            if filename == 'askreddit': continue
            log_file.write(filename + '\n') 
            month = 'RC_2019-05'
            path = SR_DATA_DIR + filename + '/' + month
            log_file.write('\tReading in textfile\n')
            data = sc.textFile(path)
            data = data.flatMap(lambda line: line.lower().split(" "))
            # TODO: tokenization step should remove punctuation from words so delete this later
            data = data.map(lambda word: regex.sub('', word))
            data = data.map(lambda word: (word, 1))
            log_file.write('\tReducing by key...\n') 
            data = data.reduceByKey(lambda a, b: a + b)
            df = sqlContext.createDataFrame(data, ['word', 'count'])
            outpath = WORD_COUNT_DIR + filename
            df.write.mode('overwrite').parquet(outpath) 
    log_file.close() 

def calculate_pmi(percent_param=0.2): 
    """
    PMI is defined as 
    log(p(word|community) / p(word)) 
    or 
    log(frequency of word in community c / frequency of word in 
    all c's we are looking at)
    as defined in Zhang et al. 2017.
    
    @output: 
        - dictionaries for each doc of word : pmi 
        - docs: list of docs in order of matrix
        - words: list of words in order of matrix
    """
    log_file = open(LOG_DIR + 'pmi.temp', 'w')
    # TODO: load json of total word counts
    docs = sorted(os.listdir(WORD_COUNT_DIR))
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        pmi_d = {}
        if os.path.isdir(WORD_COUNT_DIR + filename): 
            log_file.write(filename + '\n') 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            num_top_p = int(percent_param*len(d))
            for w in d.most_common(num_top_p): 
                pmi_d[w[0]] = d[w[0]] / float(total_counts[w[0]])
            with open(PMI_DIR + filename + '_' + str(percent_param) + '.json', 'w') as outfile: 
                sorted_d = sorted(x.items(), key=operator.itemgetter(1))
                for tup in sorted_d: 
                    outfile.write(tup[0] + '\t' + str(tup[1]) + '\n')
    log_file.close()
    
def count_overall_words_small(percent_param=0.2): 
    subreddits = ['justnomil', 'gardening', 'todayilearned', 'vegan', 'brawlstars']
    vocab = set()
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename): 
            if filename not in subreddits: continue
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            num_top_p = int(percent_param*len(d))
            for w in d.most_common(num_top_p): 
                vocab.add(w[0])
    log_file = open(LOG_DIR + 'counting_all_log.temp', 'w')
    rdd1 = sc.emptyRDD()
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename): 
            log_file.write(filename + '\n') 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            rdd2 = parquetFile.rdd.map(tuple)
            # filter to only words in our vocab
            rdd2 = rdd2.filter(lambda tup: tup[0] in vocab)
            rdd1 = rdd2.union(rdd1).reduceByKey(lambda x,y : x+y)
    df = sqlContext.createDataFrame(rdd1, ['word', 'count'])
    df.write.mode('overwrite').parquet(LOG_DIR + 'total_word_counts') 
    log_file.close()

def count_overall_words(): 
    #all_counts = Counter()
    # TODO: do only for words in a subset of 5 subreddits
    log_file = open(LOG_DIR + 'counting_all_log.temp', 'w')
    rdd1 = sc.emptyRDD()
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename): 
            log_file.write(filename + '\n') 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            rdd2 = parquetFile.rdd.map(tuple)
            rdd1 = rdd2.union(rdd1).reduceByKey(lambda x,y : x+y)
            #d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            #all_counts += d
    df = sqlContext.createDataFrame(rdd1, ['word', 'count'])
    df.write.mode('overwrite').parquet(LOG_DIR + 'total_word_counts') 
    #df.write.format('json').save(LOG_DIR + 'total_word_counts', overwrite=True)
    log_file.close()
#     with open(LOG_DIR + 'total_word_counts.json', 'w') as outfile: 
#         json.dump(all_counts, outfile)

def word_tfidf(): 
    """
    tf-idf is defined as 
    (1 + log tf)xlog_10(N/df) where N is the 
    number of documents, or subreddits. tf is
    term frequency in document and df is
    number of documents the term appears in.

    @output: 
        - matrix of doc x word tf-idf'
        - docs: list of docs in order of matrix
        - words: list of words in order of matrix
    """
    pass

def examine_outliers(): 
    """
    Look at words with highest values in each row in
    document x word matrix and see if they make sense.
    """
    pass

def main(): 
    #count_words()
    count_overall_words_small()
    #calculate_pmi()
    sc.stop()

if __name__ == '__main__':
    main()
