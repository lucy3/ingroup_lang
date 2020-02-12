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
from nltk.tokenize import word_tokenize
#from transformers import BasicTokenizer

ROOT = '/data0/lucy/ingroup_lang/'
WORD_COUNT_DIR = ROOT + 'logs/word_counts/'
LOG_DIR = ROOT + 'logs/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def save_sr_vocab(percent_param): 
    '''
    for each subreddit, save the top percent_param*100% 
    of words into a file
    '''
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename:  
            output_path = LOG_DIR + 'sr_sense_vocab/' + filename + '_' + str(percent_param*100)
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            num_top_p = int(percent_param*len(d))
            with open(output_path, 'w') as outfile: 
                for w in d.most_common(num_top_p): 
                    outfile.write(w[0].encode('utf-8', 'replace') + '\n')


def get_vocab(percent_param, N):
    '''
    This function is python 2.7 unfortunately
    '''
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
            
def comments_with_vocab(vocab_file): 
    vocab = set()
    with open(vocab_file, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split(',')
            vocab.add(contents[0])
    d = Counter()
    tokenizer = BasicTokenizer(do_lower_case=True)
    for folder in os.listdir(ROOT + 'subreddits_month/'): 
        if not os.path.isdir(ROOT + 'subreddits_month/' + folder): continue
        data = sc.textFile(ROOT + 'subreddits_month/' + folder + '/RC_sample')
        data = data.filter(lambda line: not line.startswith('USER1USER0USER'))
        data = data.map(lambda line: set(tokenizer.tokenize(line.lower())))
        data = data.map(lambda s: (len(vocab & s), 1))
        data = data.filter(lambda tup: tup[0] != 0)
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        d += Counter(data.collectAsMap())
    print("NUMBER OF COMMENTS WITH VOCAB WORDS:", sum(d.values()))
    
def get_vocab_overlap(vocab1_path, vocab2_path):
    vocab1 = set()
    vocab2 = set()
    with open(vocab1_path, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split(',')
            vocab1.add(contents[0])
    with open(vocab2_path, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split(',')
            vocab2.add(contents[0])
    print("OVERLAP between", vocab1_path, vocab2_path, "is:", len(vocab1 & vocab2))
        
def main(): 
    #get_vocab_overlap(LOG_DIR + 'vocabs/10_20', LOG_DIR + 'vocabs/20_100')
    #get_vocab_overlap(LOG_DIR + 'vocabs/10_20', LOG_DIR + 'vocabs/3_1_filtered')
    #comments_with_vocab(LOG_DIR + 'vocabs/3_1_filtered')
    get_vocab(0.03, 1)
    save_sr_vocab(0.03)
    sc.stop()

if __name__ == "__main__":
    main()
