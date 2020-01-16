"""
File for getting vocab of words we want to get senses for
"""

import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json
from collections import Counter, defaultdict
import operator
import re, string
import csv
import math

ROOT = '/data0/lucy/ingroup_lang/'
WORD_COUNT_DIR = ROOT + 'logs/word_counts/'
PMI_DIR = ROOT + 'logs/pmi/'
TFIDF_DIR = ROOT + 'logs/tfidf/'
SR_DATA_DIR = ROOT + 'subreddits3/'
LOG_DIR = ROOT + 'logs/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def get_vocab(percent_param, N): 
    vocab_map = defaultdict(list) # word : [subreddit]
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename:  
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            num_top_p = int(percent_param*len(d))
            for w in d.most_common(num_top_p): 
                vocab_map[w[0]].append(filename)
    vocab = set()
    for w in vocab_map: 
        if len(vocab_map[w]) >= N: 
            vocab.add(w)
    all_counts = Counter()
    vocab_subreddits = Counter() 
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename: 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            count_rdd = parquetFile.rdd.map(tuple)
            # filter to only words in our vocab
            count_rdd = count_rdd.filter(lambda tup: tup[0] in vocab)
            d = Counter(count_rdd.collectAsMap())
            for w in d: 
                vocab_subreddits[w] += 1
            all_counts += d
    with open(LOG_DIR + 'vocabs/' + str(int(percent_param*100)) + '_' + str(N), 'w') as outfile:
        for w in vocab: 
            outfile.write(w.encode('utf-8', 'replace') + ',' + str(all_counts[w]) + \
                         ',' + str(vocab_subreddits[w]) + '\n')

def main(): 
    get_vocab(0.2, 20)
    get_vocab(0.2, 50)
    get_vocab(0.2, 100)
    get_vocab(0.1, 20)
    sc.stop()

if __name__ == "__main__":
    main()
