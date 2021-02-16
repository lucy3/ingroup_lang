"""
This script gets the vocab of words we want to get senses for
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
import numpy as np
#from transformers import BasicTokenizer

#ROOT = '/data0/lucy/ingroup_lang/'
ROOT = '/mnt/data0/lucy/ingroup_lang/'
WORD_COUNT_DIR = ROOT + 'logs/word_counts/'
LOG_DIR = ROOT + 'logs/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def save_sr_vocab(percent_param): 
    '''
    For each subreddit, save the top percent_param*100% 
    of words into a file. This means we only induce senses
    for a subreddit's own vocab. It's possible that a word
    in the top 10% of one subreddit won't be in the top 10% of another, 
    but it'll still be included in the overall vocab. 
    '''
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename:  
            output_path = LOG_DIR + 'sr_sense_vocab/' + filename + '_' + str(percent_param*100)
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            num_top_p = int(percent_param*len(d))
            with open(output_path, 'w') as outfile: 
                for w in d.most_common(num_top_p): 
                    outfile.write(w[0] + '\n')


def get_vocab(percent_param, N):
    '''
    Get words in top percent_param of subreddits
    that appear in at least N subreddits 
    
    This outputs a file of the format
    word, number of times it occurs overall, number of subreddits it occurs in
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
            outfile.write(w + ',' + str(all_counts[w]) + \
                         ',' + str(vocab_subreddits[w]) + '\n')
            
def comments_with_vocab(vocab_file): 
    '''
    Examines the number of comments that contain words in our vocabulary
    '''
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
    '''
    This function was used during preliminary experiments
    to understand the amount of overlap between different vocabs
    formed using different parameters. 
    '''
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

def find_missing_words(): 
    '''
    This was probably used to realize that
    emojis are not very friendly to BERT and were often missing. 
    '''
    with open(LOG_DIR + 'vocabs/vocab_map.json', 'r') as infile: 
        d = json.load(infile)
    ids = {}
    for w in d: 
        ids[d[w]] = w
    docs = set()
    for filename in os.listdir(LOG_DIR + 'vocabs/docs/'): 
        if filename.startswith('.'): continue
        docs.add(int(filename))
    diff = set(ids.keys()) - docs
    for idx in diff: 
        print("MISSING", idx, ids[idx])

def approximate_num_matches(): 
    '''
    Get an approx idea of how many vocab words we'll be matching for each subreddit
    '''
    vocab = set()
    with open(LOG_DIR + 'vocabs/10_1_filtered', 'r') as infile:
        for line in infile: 
             vocab.add(line.strip())
    overlaps = []
    for filename in os.listdir(LOG_DIR + 'sr_sense_vocab/'): 
        if not filename.endswith('_10.0'): continue
        path = LOG_DIR + 'sr_sense_vocab/' + filename 
        sr_vocab = set()
        with open(path, 'r') as infile: 
            for line in infile: 
                sr_vocab.add(line.strip())
        overlaps.append(len(vocab & sr_vocab))
    print("AVERAGE NUM MATCHES", np.mean(overlaps))
        
def main(): 
    get_vocab(0.1, 1)
    save_sr_vocab(0.1)
    sc.stop()

if __name__ == "__main__":
    main()
