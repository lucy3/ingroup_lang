import os
from collections import Counter
import matplotlib
import numpy as np 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from transformers import BertTokenizer, BasicTokenizer
from pyspark.sql import SQLContext
import math
import json

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

ROOT = '/mnt/data0/lucy/ingroup_lang/'
SR_FOLDER_MONTH = ROOT + 'subreddits_month/'
LOGS = ROOT + 'logs/'

def get_comment_length():
    '''
    Get distribution of comment lengths
    ''' 
    d = Counter()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for sr in os.listdir(SR_FOLDER_MONTH): 
        data = sc.textFile(SR_FOLDER_MONTH + sr + '/RC_sample')
        data = data.filter(lambda line: not line.startswith('USER1USER0USER'))
        data = data.map(lambda line: len(tokenizer.tokenize(line)))
        data = data.map(lambda size: (math.floor(size/10.0)*10, 1))
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        output = data.collectAsMap()
        d += output
    with open(LOGS + 'comment_lengths.json', 'w') as outfile: 
        json.dump(d, outfile)

def get_num_tokens(): 
    d = 0
    tokenizer = BasicTokenizer(do_lower_case=True)
    for sr in os.listdir(SR_FOLDER_MONTH):
        data = sc.textFile(SR_FOLDER_MONTH + sr + '/RC_sample')
        data = data.filter(lambda line: not line.startswith('USER1USER0USER'))
        data = data.map(lambda line: len(tokenizer.tokenize(line)))
        d += sum(data.collect())
    print("Total number of tokens:", d)
    
def count_comments(): 
    comment_count = Counter()
    user_count = Counter()
    for sr in os.listdir(SR_FOLDER_MONTH): 
        num_lines = 0
        num_users = 0
        with open(SR_FOLDER_MONTH + sr + '/RC_sample', 'r') as infile: 
            for line in infile: 
                if line.startswith('USER1USER0USER'): 
                    num_users += 1
                else: 
                    num_lines += 1
        comment_count[sr] = num_lines
        user_count[sr] = num_users
    with open(LOGS + 'dataset_statistics_comments.txt', 'w') as outfile: 
        for tup in comment_count.most_common(): 
            outfile.write(tup[0] + '\t' + str(tup[1]) + '\t' + str(user_count[tup[0]]) + '\n')

def main(): 
    #get_comment_length()
    #count_comments()
    get_num_tokens()
    sc.stop()

if __name__ == "__main__":
    main()
