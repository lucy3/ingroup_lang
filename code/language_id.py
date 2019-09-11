import json
import time
from tqdm import tqdm
import re
import string
import os
from pyspark import SparkConf, SparkContext
from collections import Counter

DATA = '/data0/lucy/ingroup_lang/data/'
LOG_DIR = '/data0/lucy/ingroup_lang/logs/'
SUBREDDITS = DATA + 'subreddit_list.txt'
conf = SparkConf()
sc = SparkContext(conf=conf)
sc.addFile('/data0/lucy/langid.py/langid/langid.py')
import langid

reddits = set()

def get_language(line): 
    comment = json.loads(line)
    text = comment['body'].lower()
    return (comment['subreddit'].lower(), [langid.classify(text)[0]])

def subreddit_of_interest(line): 
    comment = json.loads(line)
    return 'subreddit' in comment and 'body' in comment and \
        comment['subreddit'].lower() in reddits
    
def id_langs(): 
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            reddits.add(line.strip().lower())
    global MONTH
    MONTH = 'RC_2019-05'
    path = DATA + MONTH
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    data = data.map(get_language)
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    data = data.map(lambda tup: (tup[0], Counter(tup[1]).most_common()))
    d = data.collectAsMap()
    sc.stop()
    with open(LOG_DIR + 'subreddit_langs.json', 'w') as outfile: 
        json.dump(d, outfile)

def main(): 
    id_langs()

if __name__ == '__main__':
    main()
