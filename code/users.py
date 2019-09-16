from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json
import os

ROOT = '/data0/lucy/ingroup_lang/'
DATA = ROOT + 'data/'
SR_FOLDER = ROOT + 'subreddits/'
LOG_DIR = ROOT + 'logs/'

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
    
def main(): 
    count_unique_users()
    sc.stop()

if __name__ == '__main__':
    main()
