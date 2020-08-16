# Python 2.7
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import json
import os
import csv
import math
from collections import Counter, defaultdict
from io import StringIO
import tqdm
import numpy as np

ROOT = '/mnt/data0/lucy/ingroup_lang/'
LOG_DIR = ROOT + 'logs/'
METRIC = 'bert-base'
if METRIC == 'finetuned':  
    PMI_DIR = LOG_DIR + 'finetuned_sense_pmi/'
    MAX_PMI_DIR = LOG_DIR + 'ft_max_sense_pmi/'
    SENSE_DIR = LOG_DIR + 'finetuned_senses/'
    ALL_SENSES = LOG_DIR + 'ft_total_sense_counts.json'
    SUB_TOTALS = LOG_DIR + 'ft_sr_totals.json' 
elif METRIC == 'bert-base':
    PMI_DIR = LOG_DIR + 'base_sense_pmi/'
    MAX_PMI_DIR = LOG_DIR + 'base_max_sense_pmi/'
    SENSE_DIR = LOG_DIR + 'senses/'
    ALL_SENSES = LOG_DIR + 'base_total_sense_counts.json'
    SUB_TOTALS = LOG_DIR + 'base_sr_totals.json' 
elif METRIC == 'denoised': 
    PMI_DIR = LOG_DIR + 'denoised_sense_pmi/' 
    MAX_PMI_DIR = LOG_DIR + 'dn_max_sense_pmi/'
    SENSE_DIR = LOG_DIR + 'finetuned_senses/'
    ALL_SENSES = LOG_DIR + 'dn_total_sense_counts.json'
    SUB_TOTALS = LOG_DIR + 'dn_sr_totals.json'
VOCAB_DIR = ROOT + 'logs/sr_sense_vocab/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def user_sense(line): 
    contents = line.strip().split('\t') 
    user = contents[0].split('_')[1]
    sense = contents[1] + '#####' + contents[2]
    return ((user, sense), 1) 

def count_overall_senses_denoised(): 
    '''
    The bottom fraction only contains counts totals for the top 10 subreddits
    '''
    log_file = open(LOG_DIR + 'counting_senses_log.temp', 'w') 
    all_counts = defaultdict(list) 
    for filename in sorted(os.listdir(SENSE_DIR)): 
       log_file.write(filename + '\n') 
       data = sc.textFile(SENSE_DIR + filename) 
       data = data.map(user_sense)
       data = data.reduceByKey(lambda n1, n2: 1)
       data = data.map(lambda tup: (tup[0][1], 1))
       data = data.reduceByKey(lambda n1, n2: n1 + n2)
       d = Counter(data.collectAsMap())
       for w in d: 
           all_counts[w].append(d[w])
    summed_counts = Counter()
    for w in all_counts: 
       counts = sum(sorted(all_counts[w], reverse=True)[:10])
       summed_counts[w] = counts
    with open(ALL_SENSES, 'w') as outfile: 
       json.dump(summed_counts, outfile)
    log_file.close()

def count_overall_senses(): 
    '''
    1) Count each sense once per user per subreddit. 
    2) Sum up the counts overall subreddits and save to a file
    '''
    log_file = open(LOG_DIR + 'counting_senses_log.temp', 'w') 
    all_counts = Counter()
    subreddit_totals = Counter()
    for filename in sorted(os.listdir(SENSE_DIR)):
        log_file.write(filename + '\n') 
        data = sc.textFile(SENSE_DIR + filename) 
        data = data.map(user_sense)
        data = data.reduceByKey(lambda n1, n2: 1)
        data = data.map(lambda tup: (tup[0][1], 1))
        data = data.reduceByKey(lambda n1, n2: n1 + n2)
        d = Counter(data.collectAsMap())
        subreddit_totals[filename] = sum(list(d.values()))
        all_counts += d
    with open(SUB_TOTALS, 'w') as outfile: 
        json.dump(subreddit_totals, outfile)
    with open(ALL_SENSES, 'w') as outfile: 
        json.dump(all_counts, outfile)
    log_file.close()

def calculate_pmi(): 
    '''
    PMI
    1) For each subreddit, filter to top 10% of words in that subreddit 
    2) Count each sense once per user per subreddit
    3) Calculate PMI
    '''
    log_file = open(LOG_DIR + 'sense_pmi.temp', 'w')
    with open(ALL_SENSES, 'r') as infile: 
        total_counts = json.load(infile)
    overall_total = sum(list(total_counts.values()))
    with open(SUB_TOTALS, 'r') as infile: 
        subreddit_totals = json.load(infile)
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
        pmi = Counter()
        for k in d:
            p_k_given_c = d[k] / float(subreddit_totals[filename])
            p_k = total_counts[k] / float(overall_total)
            pmi[k] = math.log(p_k_given_c / p_k)
        with open(PMI_DIR + filename + '.csv', 'w') as outfile: 
            writer = csv.writer(outfile)
            writer.writerow(['sense', 'pmi', 'count'])
            for tup in pmi.most_common(): 
                writer.writerow([tup[0], str(tup[1]), str(d[tup[0]])])
    log_file.close()

def inspect_word(word, subreddit=None): 
    '''
    looks at max sense pmi for different subreddits for one word 
    '''
    if subreddit is not None: 
        print("~~~~~ WORD:", word, "SUBREDDIT:", subreddit, "~~~~~~")
        with open(PMI_DIR + subreddit + '.csv', 'r') as infile: 
            scores = Counter()
            counts = Counter()
            reader = csv.DictReader(infile)
            for row in reader: 
                if row['sense'].split('#####')[0] == word: 
                    scores[row['sense']] = float(row['pmi'])
                    counts[row['sense']] = int(row['count'])
            top_sense, score = scores.most_common(1)[0]
            for tup in scores.most_common(): 
                print(tup[0], tup[1], counts[tup[0]])
        d = Counter()
        for filename in tqdm.tqdm(sorted(os.listdir(PMI_DIR)), leave=False): 
            with open(PMI_DIR + filename, 'r') as infile: 
                reader = csv.DictReader(infile)
                for row in reader: 
                    if row['sense'] == top_sense: 
                        d[filename] = float(row['pmi'])
        print(d.most_common()[:5])
    else: 
        d = Counter()
        for filename in tqdm.tqdm(sorted(os.listdir(PMI_DIR))): 
            scores = []
            with open(PMI_DIR + filename, 'r') as infile: 
                reader = csv.DictReader(infile)
                for row in reader: 
                    if row['sense'].split('#####')[0] == word: 
                        scores.append(float(row['pmi']))
            if scores != []: 
                d[filename.replace('.csv', '')] = max(scores)
        print(d.most_common())
        print(np.mean(list(d.values())))
        print(np.var(list(d.values())))
    
def calc_max_pmi(): 
    '''
    For each file in PMI_DIR, get max sense pmi
    '''
    for filename in tqdm.tqdm(sorted(os.listdir(PMI_DIR))): 
        scores = defaultdict(list) # word : [list of scores]
        counts = Counter() # word : count
        with open(PMI_DIR + filename, 'r') as infile: 
            reader = csv.DictReader(infile)
            for row in reader: 
                word = row['sense'].split('#####')[0]
                scores[word].append(float(row['pmi']))
                counts[word] += int(row['count'])
        with open(MAX_PMI_DIR + filename, 'w') as outfile: 
            writer = csv.writer(outfile)
            writer.writerow(['word', 'max_pmi', 'count'])
            for word in scores: 
                writer.writerow([word, str(max(scores[word])), str(counts[word])])

def main(): 
    count_overall_senses()
    calculate_pmi()
    
    #inspect_word('cubes', 'azurelane')
    #inspect_word('hesitation', 'sekiro')
    #inspect_word('granted', 'themonkeyspaw')
    #inspect_word('hunters', 'borderlands')
    #inspect_word('island', 'loveislandtv')
    #inspect_word('monk', 'sekiro')
    #inspect_word('labs', 'crashbandicoot')
    #inspect_word('gb', 'forhonor')
    #inspect_word('abundance', 'edh')
    #inspect_word('tags', 'music') 
    #inspect_word('bowls')
    #inspect_word('curry')
    #inspect_word('pm')
    #inspect_word('associates')
    #inspect_word('spark')

    calc_max_pmi()
    sc.stop()

if __name__ == '__main__':
    main()
