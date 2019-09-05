from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json

DATA = '/data0/lucy/ingroup_vocab/data/'
LOG_DIR = '/data0/lucy/ingroup_vocab/logs/'
SUBREDDITS = DATA + 'subreddit_list.txt'

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
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            reddits.add(line.strip().lower())
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
    df.write.format('com.databricks.spark.csv').mode('overwrite').options(header='true').save(outpath)
    
def main(): 
    count_unique_users()
    sc.stop()

if __name__ == '__main__':
    main()