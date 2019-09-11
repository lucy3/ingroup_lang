"""
Functions for organizing or reformatting 
Reddit data.

Python 2.7
"""
import json
import time
from tqdm import tqdm
import re
import string
import os
from pyspark import SparkConf, SparkContext
from collections import Counter

DATA = '/global/scratch/lucy3_li/ingroup_lang/data/'
SR_FOLDER = '/global/scratch/lucy3_li/ingroup_lang/subreddits/'
SUBREDDITS = DATA + 'subreddit_list.txt'
REMOVED_SRS = DATA + 'non_english_sr.txt'

conf = SparkConf()
sc = SparkContext(conf=conf)
reddits = set()

def get_comment(line): 
    comment = json.loads(line)
    text = comment['body']
    # remove urls
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                  '', text, flags=re.MULTILINE)
    # remove new lines
    text = text.replace('\n', ' ').replace('\r', ' ')
    return (comment['subreddit'].lower(), text)
    
def subreddit_of_interest(line): 
    comment = json.loads(line)
    return 'subreddit' in comment and 'body' in comment and \
        comment['subreddit'].lower() in reddits
    
def save_doc(item): 
    if item[0] is not None:
        path = SR_FOLDER + item[0] + '/'
        with open(path + MONTH, 'w') as file:
            file.write(item[1].encode('utf-8', 'replace'))
            
def get_subreddit(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment and \
	    comment['body'] != '[deleted]' and comment['body'] != '[removed]': 
        return (comment['subreddit'].lower(), 1)
    else: 
        return (None, 0)

def get_top_subreddits(n=300): 
    '''
    Get the top n subreddits by number
    of comments. 
    Takes ~30 min for 1 month on redwood
    using --master 'local[*]'
    '''
    path = DATA + 'RC_2019-05'
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.map(get_subreddit)
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    data = data.collectAsMap()
    sr_counts = Counter(data)
    with open(SUBREDDITS, 'w') as outfile: 
        for sr in sr_counts.most_common(n): 
            outfile.write(sr[0] + '\n') 
    
def create_subreddit_docs(): 
    '''
    Create a document for each subreddit by month
    '''
    non_english_reddits = set()
    with open(REMOVED_SRS, 'r') as inputfile: 
        for line in inputfile: 
            non_english_reddits.add(line.strip().lower())
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            sr = line.strip().lower()
            if sr not in non_english_reddits: 
                reddits.add(sr)
    # create output folders
    for sr in reddits: 
        path = SR_FOLDER + sr + '/'
        if not os.path.exists(path): 
            os.makedirs(path)
    global MONTH
    MONTH = 'RC_2019-05'
    #path = DATA + MONTH
    path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    data = data.map(get_comment)
    data = data.reduceByKey(lambda n1, n2: n1 + '\n' + n2)
    data = data.foreach(save_doc)

def main(): 
    #get_top_subreddits(n=500)
    create_subreddit_docs()
    sc.stop()

if __name__ == '__main__':
    main()
