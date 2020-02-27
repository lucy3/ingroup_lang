from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json
import os
import csv
from collections import Counter

ROOT = '/mnt/data0/lucy/ingroup_lang/'
LOG_DIR = ROOT + 'logs/' 
PMI_DIR = LOG_DIR + '/finetuned_sense_pmi/'
SENSE_DIR = LOG_DIR + '/finetuned_senses/'
VOCAB_DIR = LOG_DIR + '/sr_sense_vocab/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def user_sense(line): 
    contents = line.strip().split('\t') 
    user = contents[0].split('_')[1]
    sense = contents[1] + '#####' + contents[2]
    return ((user, sense), 1) 

def count_overall_senses(): 
    '''
    1) Count each sense once per user per subreddit. 
    2) Sum up the counts overall subreddits and save to a file
    '''
    log_file = open(LOG_DIR + 'counting_senses_log.temp', 'w') 
    all_counts = Counter()
    for filename in sorted(os.listdir(SENSE_DIR)):
        log_file.write(filename + '\n') 
        data = sc.textFile(SENSE_DIR + filename) 
        data = data.map(user_sense)
        data = data.reduceByKey(lambda n1, n2: 1)
        data = data.map(lambda tup: (tup[0][1], 1))
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        d = Counter(data.collectAsMap())
        all_counts += d
    with open(LOG_DIR + 'total_sense_counts.json', 'w') as outfile: 
        json.dump(all_counts, outfile)
    log_file.close()

def calculate_pmi(): 
    '''
    PMI is defined as 
    frequency of sense in community c / frequency of sense in all communities
    1) For each subreddit, filter to top 10% of words in that subreddit 
    2) Count each sense once per user per subreddit
    3) Calculate PMI
    '''
    log_file = open(LOG_DIR + 'sense_pmi.temp', 'w')
    with open(LOG_DIR + 'total_sense_counts.json', 'r') as infile: 
        total_counts = json.load(infile)
    for filename in sorted(os.listdir(SENSE_DIR)): 
        log_file.write(filename + '\n') 
        data = sc.textFile(SENSE_DIR + filename)
        vocab = set()
        with open(VOCAB_DIR + filename + '_10.0', 'r') as infile: 
            for line in infile: 
                vocab.add(line.strip())
        data = data.filter(lambda line: line.split('\t')[1] in vocab)
        data = data.map(user_sense)
        data = data.reduceByKey(lambda n1, n2: 1)
        data = data.map(lambda tup: (tup[0][1], 1))
        data = data.reduceByKey(lambda n1, n2: n1 + n2) 
        d = Counter(data.collectAsMap())
        with open(PMI_DIR + filename + '.csv', 'w') as outfile: 
            writer = csv.writer(outfile)
            writer.writerow(['sense', 'pmi', 'count'])
            for tup in d.most_common(): 
                pmi = tup[1] / float(total_counts[tup[0]])
                writer.writerow([tup[0], str(pmi), str(tup[1])])
    log_file.close()

def main(): 
    count_overall_senses()
    calculate_pmi()
    sc.stop()

if __name__ == '__main__':
    main()
