"""
Comparison of our measurements
of a word belonging to a sociolect
to actual subreddit glossaries.
"""
import csv
from collections import Counter, defaultdict
import numpy as np
import corenlp
import os 

ROOT = '/data0/lucy/ingroup_lang/'
SR_LIST = ROOT + 'data/glossary_list.csv'
TERMS = ROOT + 'data/glossaries.csv'
SCORE_LOG = ROOT + 'logs/glossary_eval/'
PMI_PATH = ROOT + 'logs/pmi/'
TFIDF_PATH = ROOT + 'logs/tfidf/'

def basic_stats(): 
    '''
    Number of subreddits, number of terms per subreddit
    
    Check that all subreddits with links have glossary standardized terms
    and that subreddit names match up in these files. 
    '''
    sr_list = []
    with open(SR_LIST, 'r') as infile: 
        reader = csv.DictReader(infile, delimiter=',')
        for row in reader:
            if row['glossary'].strip() != '': 
                sr_list.append(row['subreddit_name'])
    sr_list = sorted(sr_list)
    sr_set = set()
    sr_count = Counter()
    with open(TERMS, 'r') as infile: 
        reader = csv.DictReader(infile, delimiter=',')
        for row in reader:
            sr_set.add(row['subreddit'].strip())
            sr_count[row['subreddit']] += 1
    assert sorted(sr_list) == sorted(sr_set)
    print("Number of subreddits:", len(sr_set))
    print("Average number of terms per subreddit:", np.mean(sr_count.values()))
    print("Min number of terms:", np.min(sr_count.values()))
    print("Max number of terms:", np.max(sr_count.values()))

def get_sr2terms(): 
    sr2terms = defaultdict(list)
    with corenlp.CoreNLPClient(annotators="tokenize".split()) as client:
        with open(TERMS, 'r') as infile: 
            reader = csv.DictReader(infile, delimiter=',')
            for row in reader:
                term = row['term'].lower()
                ann = client.annotate(term)
                for tok in ann.sentencelessToken:
                    sr2terms[row['subreddit']].append(tok.word)
    return sr2terms
 
        
def compute_fscore(sr2terms, metric, score_cutoff, count_cutoff=0): 
    '''
    For multiword expressions, we tokenize these using
    corenlp tokenizer. 
    
    recall - out of all glossary words, what fraction do we have?
    precision - out of all words we have listed, how many are glossary words?
    fscore
    
    we calculate these values for each subreddit and average them.
    '''
    if metric == 'tfidf': 
        inpath = TFIDF_PATH
    elif metric == 'pmi': 
        inpath = PMI_PATH
    else: 
        raise ValueError("Not implemented yet!")
    rs = []
    ps = []
    fs = []
    print("Score cutoff =", score_cutoff, "| Count cutoff =", count_cutoff)
    log_file = open(SCORE_LOG + metric + '_' + str(score_cutoff) + '_' + str(count_cutoff), 'w')
    log_file.write("subreddit recall precision f1\n")
    for subreddit_file in os.listdir(inpath): 
        if not subreddit_file.endswith('_0.2.csv'): continue
        subreddit_name = subreddit_file.replace('_0.2.csv', '')
        if subreddit_name not in sr2terms: continue
        sociolect_words = set() 
        with open(inpath + subreddit_file, 'r') as infile: 
            reader = csv.DictReader(infile, delimiter=',')
            for row in reader: 
                if float(row[metric]) > score_cutoff and int(row['count']) > count_cutoff: 
                    sociolect_words.add(row['word'])
        num_gloss_words = len(sr2terms[subreddit_name])
        num_our_list = len(sociolect_words)
        overlap = len(set(sr2terms[subreddit_name]) & sociolect_words)
        recall = overlap / float(num_gloss_words)
        if num_our_list > 0: 
            precision = overlap / float(num_our_list)
        else: 
            precision = 0
        if (recall + precision) > 0: 
            f1 = 2*precision*recall/(recall + precision)
        else: 
            f1 = 0
        log_file.write(subreddit_name + ' ' + str(recall) + ' ' + str(precision) + ' ' + str(f1) + '\n') 
        rs.append(recall)
        ps.append(precision)
        fs.append(f1)
    r = np.mean(rs)
    p = np.mean(ps)
    f = np.mean(fs)
    log_file.close()
    print("Recall:", r)
    print("Precision:", p)
    print("F1 score:", f)
    return r, p, f

def find_best_parameters():
    sr2terms = get_sr2terms() 
    best = None
    best_score = 0
    metric = 'tfidf'
    for cutoff in [0, 1, 2, 3, 4, 5, 6, 7, 8]: 
        r, p, f = compute_fscore(sr2terms, metric, cutoff)
        if f > best_score: 
            best_score = f
            best = (metric, cutoff, r, p, f)
    print("BEST PARAMS (metric, cutoff, recall, precision, f1 score):")
    print(best)

    best = None
    best_score = 0
    metric = 'pmi'
    for cutoff in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, \
                  0.8, 0.9]: 
        r, p, f = compute_fscore(sr2terms, metric, cutoff)
        if f > best_score: 
            best_score = f
            best = (metric, cutoff, r, p, f)
    print("BEST PARAMS (metric, cutoff, recall, precision, f1 score):")
    print(best)

def main(): 
    #basic_stats()
    find_best_parameters()

if __name__ == '__main__':
    main()
