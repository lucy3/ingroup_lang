from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json
import os

ROOT = '/data0/lucy/ingroup_lang/'
DATA = ROOT + 'data/'
SR_FOLDER = ROOT + 'subreddits/'
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
    global MONTH
    MONTH = 'RC_2019-05'
    path = DATA + MONTH
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
    global MONTH
    MONTH = 'RC_2019-05'
    path = DATA + MONTH
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    subreddits = data.map(get_subreddit)
    # total num of comments per subreddit
    subreddits = subreddits.reduceByKey(lambda n1, n2: n1 + n2) 
    total_com = subreddits.collectAsMap()
    outfile = open(LOG_DIR + 'commentor_activity', 'w')
    # TODO: need to read commentor_counts as a pandas dataframe 
    commentor_path = LOG_DIR + '/commentor_counts/part-00000-f83d5d87-c50d-4d5a-a560-e978e85e0af8-c000.csv'
    outfile.write('subreddit,activity\n')
    with open(commentor_path, 'r') as infile: 
        for line in infile: 
            if line.startswith('subreddit,'): continue
            contents = line.strip().split(',')
            sr = contents[0]
            c = float(contents[1])
            outfile.write(sr + ',' + str(total_com[sr] / c) + '\n')
    outfile.close()

def main(): 
    #count_unique_users()
    user_activity()
    sc.stop()

if __name__ == '__main__':
    main()
