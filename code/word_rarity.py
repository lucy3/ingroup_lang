# get word counts in each subreddit 
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

WORD_COUNT_DIR = '/data0/lucy/ingroup_vocab/logs/word_counts/'
SR_DATA_DIR = '/data0/lucy/ingroup_vocab/data/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def count_words(): 
    """
    Get word counts in each subreddit
    """
    for filename in os.listdir(SR_DATA_DIR): 
        if os.path.isdir(SR_DATA_DIR + filename): 
            month = 'RC_2019-05'
            path = SR_DATA_DIR + filename + '/' + month
            data = sc.textFile(path)
            data = data.flatMap(lambda line: line.split(" "))
            data = data.map(lambda word: (word, 1))
            data = data.reduceByKey(lambda a, b: a + b)
            df = sqlContext.createDataFrame(data, ['word', 'count'])
            outpath = WORD_COUNT_DIR + filename
            df.write.format('com.databricks.spark.csv').mode('overwrite').options(header='true').save(outpath)

def pmi_matrix(): 
    """
    PMI is defined as 
    log(p(word|community) / p(word)) 
    or 
    log(frequency of word in community c / frequency of word in 
    all c's we are looking at)
    as defined in Zhang et al. 2017.
    
    @output: 
        - matrix of doc x word pmi 
        - docs: list of docs in order of matrix
        - words: list of words in order of matrix
    """
    # load word counts
    # get total word counts
    # get sorted vocab 
    # get sorted subreddit list
    # get matrix of counts
    # normalize counts by total word counts
    # save matrix as pickle
    pass

def word_tfidf(): 
    """
    tf-idf is defined as 
    (1 + log tf)xlog_10(N/df) where N is the 
    number of documents, or subreddits. tf is
    term frequency in document and df is
    number of documents the term appears in.

    @output: 
        - matrix of doc x word tf-idf'
        - docs: list of docs in order of matrix
        - words: list of words in order of matrix
    """
    pass

def examine_outliers(): 
    """
    Look at words with highest values in each row in
    document x word matrix and see if they make sense.
    """
    pass

def main(): 
    count_words()
    sc.stop()

if __name__ == '__main__':
    main()
