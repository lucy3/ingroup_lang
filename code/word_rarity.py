# get word counts in each subreddit 
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json
from collections import Counter
import operator
import re, string
import csv
import math

#ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
ROOT = '/data0/lucy/ingroup_lang/'
WORD_COUNT_DIR = ROOT + 'logs/word_counts/'
PMI_DIR = ROOT + 'logs/pmi/'
TFIDF_DIR = ROOT + 'logs/tfidf/'
SR_DATA_DIR = ROOT + 'subreddits4/'
LOG_DIR = ROOT + 'logs/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

regex = re.compile('[%s]' % re.escape(string.punctuation))

def count_words(): 
    """
    Get word counts in each subreddit
    """
    log_file = open(LOG_DIR + 'counting_log.temp', 'w')
    for filename in os.listdir(SR_DATA_DIR):  
        log_file.write(filename + '\n') 
        path = SR_DATA_DIR + filename + '/RC_2019-05'
        log_file.write('\tReading in textfile\n')
        data = sc.textFile(path)
        data = data.filter(lambda line: line.strip() != '')
        data = data.map(lambda line: (line.strip(), 1))
        log_file.write('\tReducing by key...\n') 
        data = data.reduceByKey(lambda a, b: a + b)
        df = sqlContext.createDataFrame(data, ['word', 'count'])
        outpath = WORD_COUNT_DIR + filename
        df.write.mode('overwrite').parquet(outpath) 
    log_file.close()

def merge_counts(): 
    """
    Merge counts for large files. 
    """
    large_files = ['askreddit', 'amitheasshole', 'politics']
    log_file = open(LOG_DIR + 'merge_counts.temp', 'w')
    for sr in large_files: 
        total_counts = Counter()
        for filename in os.listdir(WORD_COUNT_DIR): 
            if filename.startswith(sr + '@'): 
                log_file.write(filename + '\n')
                parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
                d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
                total_counts += d
        rdd = sc.parallelize(list(total_counts.iteritems()))
        df = sqlContext.createDataFrame(rdd, ['word', 'count'])
        df.write.mode('overwrite').parquet(WORD_COUNT_DIR + sr) 
    log_file.close()

def calculate_pmi(percent_param=0.2): 
    """
    PMI is defined as 
    log(p(word|community) / p(word)) 
    or 
    log(frequency of word in community c / frequency of word in 
    all c's we are looking at)
    as defined in Zhang et al. 2017.
    """
    log_file = open(LOG_DIR + 'pmi.temp', 'w')
    with open(LOG_DIR + 'total_word_counts.json', 'r') as infile: 
        total_counts = json.load(infile)
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        pmi_d = {}
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename: 
            log_file.write(filename + '\n') 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            num_top_p = int(percent_param*len(d))
            for w in d.most_common(num_top_p): 
                pmi_d[w[0]] = d[w[0]] / float(total_counts[w[0]])
            new_filename = filename.replace('.txt', '')
            with open(PMI_DIR + new_filename + '_' + str(percent_param) + '.csv', 'w') as outfile: 
                sorted_d = sorted(pmi_d.items(), key=operator.itemgetter(1))
                writer = csv.writer(outfile)
                writer.writerow(['word', 'pmi', 'count'])
                for tup in sorted_d: 
                    writer.writerow([tup[0].encode('utf-8', 'replace'), str(tup[1]), str(d[tup[0]])])
    log_file.write("DONE\n")
    log_file.close()
    
def count_overall_words(percent_param=0.2): 
    vocab = set()
    log_file = open(LOG_DIR + 'counting_all_log.temp', 'w') 
    log_file.write("Getting vocab...\n")
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename: 
            log_file.write(filename + '\n') 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            num_top_p = int(percent_param*len(d))
            for w in d.most_common(num_top_p): 
                vocab.add(w[0])
    log_file.write("Vocab size:" + str(len(vocab)) + "\n")
    all_counts = Counter()
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename: 
            log_file.write(filename + '\n') 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            count_rdd = parquetFile.rdd.map(tuple)
            # filter to only words in our vocab
            count_rdd = count_rdd.filter(lambda tup: tup[0] in vocab)
            d = Counter(count_rdd.collectAsMap())
            all_counts += d
    with open(LOG_DIR + 'total_word_counts.json', 'w') as outfile: 
        json.dump(all_counts, outfile)
    log_file.write("DONE\n")
    log_file.close()
    
def count_document_freq(percent_param=0.2):
    vocab = set()
    log_file = open(LOG_DIR + 'counting_df_log.temp', 'w') 
    log_file.write("Getting vocab...\n")
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename: 
            log_file.write(filename + '\n') 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            num_top_p = int(percent_param*len(d))
            for w in d.most_common(num_top_p): 
                vocab.add(w[0])
    log_file.write("Vocab size:" + str(len(vocab)) + "\n")
    all_counts = Counter()
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename: 
            log_file.write(filename + '\n') 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            count_rdd = parquetFile.rdd.map(tuple)
            # filter to only words in our vocab
            count_rdd = count_rdd.filter(lambda tup: tup[0] in vocab)
            count_rdd = count_rdd.map(lambda tup: (tup[0], 1))
            d = Counter(count_rdd.collectAsMap())
            all_counts += d
    with open(LOG_DIR + 'doc_freqs.json', 'w') as outfile: 
        json.dump(all_counts, outfile)
    log_file.write("DONE\n") 
    log_file.close()
    
def word_tfidf(percent_param=0.2): 
    """
    tf-idf is defined as 
    (1 + log tf)xlog_10(N/df) where N is the 
    number of documents, or subreddits. tf is
    term frequency in document and df is
    number of documents the term appears in.
    """
    log_file = open(LOG_DIR + 'tfidf.temp', 'w')
    # load document frequency of words
    with open(LOG_DIR + 'doc_freqs.json', 'r') as infile: 
        doc_freqs = json.load(infile)
    docs = sorted(os.listdir(SR_DATA_DIR))
    N = float(len(docs))
    for filename in sorted(os.listdir(WORD_COUNT_DIR)): 
        tfidf_d = {}
        if os.path.isdir(WORD_COUNT_DIR + filename) and '@' not in filename: 
            log_file.write(filename + '\n') 
            parquetFile = sqlContext.read.parquet(WORD_COUNT_DIR + filename + '/')
            d = Counter(parquetFile.toPandas().set_index('word').to_dict()['count'])
            num_top_p = int(percent_param*len(d))
            for w in d.most_common(num_top_p): 
                tfidf_d[w[0]] = (1.0 + math.log(d[w[0]], 10))*math.log(N/doc_freqs[w[0]], 10)
            new_filename = filename.replace('.txt', '')
            with open(TFIDF_DIR + new_filename + '_' + str(percent_param) + '.csv', 'w') as outfile: 
                sorted_d = sorted(tfidf_d.items(), key=operator.itemgetter(1))
                writer = csv.writer(outfile)
                writer.writerow(['word', 'tfidf', 'count'])
                for tup in sorted_d: 
                    writer.writerow([tup[0].encode('utf-8', 'replace'), str(tup[1]), str(d[tup[0]])])
    log_file.write("DONE\n")
    log_file.close()

def niche_disem(percent_param=0.2):
    """
    Altmann's metric, but for subreddits instead of users
    
    """
    pass

def main(): 
    #count_words()
    #count_overall_words()
    #count_document_freq()
    #calculate_pmi()
    word_tfidf()
    sc.stop()

if __name__ == '__main__':
    main()
