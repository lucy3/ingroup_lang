from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json
import os

ROOT = '/mnt/data0/lucy/ingroup_lang/'
DATA = ROOT + 'data/'
SR_FOLDER = ROOT + 'subreddits_month/'
LOG_DIR = ROOT + 'logs/'
SUBREDDITS = DATA + 'subreddit_list.txt'
REMOVED_SRS = DATA + 'non_english_sr.txt'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
reddits = set()

def subreddit_of_interest(line): 
    comment = json.loads(line)
    return 'subreddit' in comment and 'body' in comment and \
        comment['subreddit'].lower() in reddits

def subreddit_of_interest_posts(line): 
    post = json.loads(line)
    return 'subreddit' in post and post['subreddit'].lower() in reddits

def get_user(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment: 
        return (comment["author"].lower(), comment['subreddit'].lower())
    else: 
        return (None, None)
    
def count_unique_users(): 
    for folder_name in os.listdir(SR_FOLDER): 
        if os.path.isdir(SR_FOLDER + folder_name): 
            reddits.add(folder_name)
    path = DATA + 'RC_all'
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    data = data.map(get_user)
    data = data.distinct()
    data = data.map(lambda x: (x[1], 1)).reduceByKey(lambda n1, n2: n1 + n2)
    df = sqlContext.createDataFrame(data, ['subreddit', 'num_commentors'])
    outpath = LOG_DIR + 'commentor_counts'
    df.coalesce(1).write.format('com.databricks.spark.csv').mode('overwrite').options(header='true').save(outpath)
    
def get_subreddit(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment and \
        comment['body'].strip() != '[deleted]' and comment['body'].strip() != '[removed]': 
        return (comment['subreddit'].lower(), 1)
    else: 
        return (None, 0)
    
def user_activity(): 
    """
    Where activity is calculated as 
    total comments / num of commentors
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
    path = DATA + 'RC_all'
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    subreddits = data.map(get_subreddit)
    # total num of comments per subreddit
    subreddits = subreddits.reduceByKey(lambda n1, n2: n1 + n2) 
    total_com = subreddits.collectAsMap()
    outfile = open(LOG_DIR + 'commentor_activity', 'w')
    commentor_path = LOG_DIR + 'commentor_counts/part-00000-64b1d705-9cf8-4a54-9c4d-598e5bf9085f-c000.csv'
    outfile.write('subreddit,activity\n')
    with open(commentor_path, 'r') as infile: 
        for line in infile: 
            if line.startswith('subreddit,'): continue
            contents = line.strip().split(',')
            sr = contents[0]
            c = float(contents[1])
            outfile.write(sr + ',' + str(total_com[sr] / c) + '\n')
    outfile.close()

def get_subscribers_info(line): 
    post = json.loads(line)
    if 'subreddit' in post: 
        return (post['subreddit'].lower(), [(post['created_utc'], post['subreddit_subscribers'])])
    else: 
        return (None, [(0, 0)])

def get_subscribers(tup): 
    sr = tup[0]
    max_time = 0
    sub_count = 0
    for time_sub in tup[1]: 
        if time_sub[0] > max_time: 
            max_time = time_sub[0]
            sub_count = time_sub[1]
    return (sr, sub_count)

def count_subscribers():
    for folder_name in os.listdir(SR_FOLDER): 
       if os.path.isdir(SR_FOLDER + folder_name): 
           reddits.add(folder_name)
    path = DATA + 'RS_2019-06'
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest_posts)
    subreddits = data.map(get_subscribers_info)
    subreddits = subreddits.reduceByKey(lambda n1, n2: n1 + n2)
    subreddits = subreddits.map(get_subscribers)
    subreddits = subreddits.collectAsMap()

    outfile = open(LOG_DIR + 'subscribers', 'w')
    commentor_path = LOG_DIR + 'commentor_counts/part-00000-64b1d705-9cf8-4a54-9c4d-598e5bf9085f-c000.csv'
    outfile.write('subreddit,num_subs\n')
    for sr in subreddits: 
        outfile.write(sr + ',' + str(subreddits[sr]) + '\n')
    outfile.close()
    
    outfile = open(LOG_DIR + 'subscribers_ratio', 'w') 
    outfile.write('subreddit,sub_ratio\n')
    with open(commentor_path, 'r') as infile: 
        for line in infile: 
            if line.startswith('subreddit,'): continue
            contents = line.strip().split(',')
            sr = contents[0]
            c = float(contents[1])
            outfile.write(sr + ',' + str(subreddits[sr] / float(c)) + '\n')
    outfile.close()


def main(): 
    #count_unique_users()
    #user_activity()
    count_subscribers()
    sc.stop()

if __name__ == '__main__':
    main()
