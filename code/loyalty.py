"""
Based on "Loyalty in Online Communities" by Hamilton et al. 2017
"""

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json
import os
from collections import Counter, defaultdict

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
reddits = set()

ROOT = '/data0/lucy/ingroup_lang/'
DATA = ROOT + 'data/'
LOG_DIR = ROOT + 'logs/'
POSTS = DATA + 'RS_2019-05'
COMMENTS = DATA + 'RC_2019-05'
SUBREDDITS = DATA + 'subreddit_list.txt'
REMOVED_SRS = DATA + 'non_english_sr.txt'

def subreddit_of_interest(line): 
    comment = json.loads(line)
    return comment['subreddit'].lower() in reddits

def extract_id(line): 
    comment = json.loads(line)
    return (comment['subreddit'].lower(), [comment['id'].lower()])

def get_post_ids(): 
    """
    This step wasn't actually needed.
    """
    non_english_reddits = set()
    with open(REMOVED_SRS, 'r') as inputfile: 
        for line in inputfile: 
            non_english_reddits.add(line.strip().lower())
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            sr = line.strip().lower()
            if sr not in non_english_reddits: 
                reddits.add(sr)
    data = sc.textFile(POSTS)
    data = data.filter(subreddit_of_interest)
    data = data.map(extract_id)
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    sr_posts = data.collectAsMap()
    with open(ROOT + 'logs/sr_post_ids.json', 'w') as outfile: 
        json.dump(sr_posts, outfile)
        
def user_subreddit(line): 
    comment = json.loads(line)
    return (comment['author'].lower(), [comment['subreddit'].lower()])
        
def get_user_subreddits(): 
    non_english_reddits = set()
    with open(REMOVED_SRS, 'r') as inputfile: 
        for line in inputfile: 
            non_english_reddits.add(line.strip().lower())
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            sr = line.strip().lower()
            if sr not in non_english_reddits: 
                reddits.add(sr)
    data = sc.textFile(COMMENTS)
    data = data.filter(subreddit_of_interest)
    # get first level comments only
    data = data.filter(lambda line: json.loads(line)['link_id'] == json.loads(line)['parent_id'])
    data = data.map(user_subreddit)
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    # only users who post >= 10 times
    data = data.filter(lambda tup: len(tup[1]) >= 10)
    user_sr = data.collectAsMap()
    with open(ROOT + 'logs/user_sr.json', 'w') as outfile: 
        json.dump(user_sr, outfile)
        
def calculate_loyalty(threshold=0.5): 
    with open(ROOT + 'logs/user_sr.json', 'r') as infile: 
        user_sr = json.load(infile)
    loyal_users = defaultdict(set)
    for user in user_sr: 
        mc = Counter(user_sr[user]).most_common()
        for i, tup in enumerate(mc): 
            if float(tup[1]) / len(user_sr[user]) >= threshold: 
                loyal_users[tup[0]].add(user)
            else: 
                break
    all_users = defaultdict(set)
    for user in user_sr: 
        srs = set(user_sr[user])
        for sr in srs: 
            all_users[sr].add(user)
    outfile = open(LOG_DIR + 'commentor_loyalty_'+str(int(threshold*100)), 'w')
    outfile.write('subreddit,loyalty\n')
    for sr in all_users: 
        prop = float(len(loyal_users[sr])) / len(all_users[sr])
        outfile.write(sr + ',' + str(prop) + '\n')
    outfile.close()

def main(): 
    #count_unique_users()
    #get_post_ids()
    #get_user_subreddits()
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]: 
        calculate_loyalty(threshold)
    sc.stop()

if __name__ == '__main__':
    main()
