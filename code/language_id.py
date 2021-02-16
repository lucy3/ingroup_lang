'''
We get the language of every comment in a subreddit. 
'''

import json
import time
from tqdm import tqdm
import re
import string
import os
from pyspark import SparkConf, SparkContext
from collections import Counter

ROOT = '/data0/lucy/ingroup_lang/'
DATA = ROOT + 'data/'
LOG_DIR = ROOT + 'logs/'
SUBREDDITS = DATA + 'subreddit_list.txt'
SR_FOLDER_MONTH = ROOT + 'subreddits_month/'
conf = SparkConf()
sc = SparkContext(conf=conf)
sc.addFile('/data0/lucy/langid.py/langid/langid.py')
import langid

reddits = set()

def get_language(text): 
    return langid.classify(text)[0]

def id_langs():
    lang_dict = {}
    log_file = open(LOG_DIR + 'language_id.temp', 'w')
    for sr in os.listdir(SR_FOLDER_MONTH): 
        log_file.write(sr + '\n') 
        path = SR_FOLDER_MONTH + sr + '/RC_sample'
        data = sc.textFile(path)
        data = data.filter(lambda line: not line.startswith('@@#USER#@@_'))
        data = data.map(get_language)
        langs = Counter(data.collect())
        lang_dict[sr] = langs.most_common()
    sc.stop()
    log_file.write("dumping....\n")
    with open(LOG_DIR + 'subreddit_langs.json', 'w') as outfile: 
        json.dump(lang_dict, outfile)
    log_file.close()

def main(): 
    id_langs()

if __name__ == '__main__':
    main()
