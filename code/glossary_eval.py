"""
Several functions looking at glossary words,
whether they appear in the dataset, 
whether they have scores computed for them
"""
import csv
from collections import Counter, defaultdict
import numpy as np
import os 
from transformers import BasicTokenizer
import matplotlib.pyplot as plt
import re
import json

ROOT = '/mnt/data0/lucy/ingroup_lang/'
SR_LIST = ROOT + 'data/glossary_list.csv'
TERMS = ROOT + 'data/glossaries.csv'
WEBPAGES = ROOT + 'data/glossaries/'
SCORE_LOG = ROOT + 'logs/glossary_eval/'
PMI_PATH = ROOT + 'logs/pmi/'
TFIDF_PATH = ROOT + 'logs/tfidf/'
SENSEPMI_PATH = ROOT + 'logs/ft_max_sense_pmi/'
BASE_SENSEPMI_PATH = ROOT + 'logs/base_max_sense_pmi/'
DN_SENSEPMI_PATH = ROOT + 'logs/dn_max_sense_pmi/'

def basic_stats(): 
    '''
    Number of subreddits, number of terms per subreddit
    
    Check that all subreddits with links have glossary standardized terms
    and that subreddit names match up in these files. 
    '''
    tokenizer = BasicTokenizer(do_lower_case=True)
    sr_list = []
    with open(SR_LIST, 'r') as infile: 
        reader = csv.DictReader(infile, delimiter=',')
        for row in reader:
            if row['glossary'].strip() != '': 
                sr_list.append(row['subreddit_name'])
    sr_list = sorted(sr_list)
    sr_set = set()
    sr_terms = defaultdict(set)
    mwe_set = defaultdict(set)
    total_terms = 0
    num_mwes = 0
    with open(TERMS, 'r') as infile: 
        reader = csv.DictReader(infile, delimiter=',')
        for row in reader:
            sr_set.add(row['subreddit'].strip())
            term = row['term'].strip().lower()
            sr_terms[row['subreddit']].add(term)
            term_len = len(tokenizer.tokenize(term))
            if term_len > 1: 
                num_mwes += 1
                mwe_set[term_len].add(term)
    assert sorted(sr_list) == sorted(sr_set)
    sr_count = Counter()
    for sr in sr_terms: sr_count[sr] = len(sr_terms[sr])
    mwe_count = Counter()
    for term_len in mwe_set: mwe_count[term_len] = len(mwe_set[term_len])
    print("Number of subreddits:", len(sr_set))
    print("Average number of terms per subreddit:", np.mean(list(sr_count.values())))
    print("Min number of terms:", np.min(list(sr_count.values())))
    print("Max number of terms:", np.max(list(sr_count.values())))
    print("Total number of terms:", sum(list(sr_count.values())))
    print("Total number of non-unique mwes:", num_mwes)
    print("Total number of unique mwes:", sum(list(mwe_count.values())))
    for term_len in sorted(mwe_count.keys()): 
        print(term_len, mwe_count[term_len])

def get_sr2terms_no_mwes(): 
    tokenizer = BasicTokenizer(do_lower_case=True)
    sr2terms = defaultdict(list)
    num_mwes = 0
    with open(TERMS, 'r') as infile: 
        reader = csv.DictReader(infile, delimiter=',')
        for row in reader:
            term = row['term'].strip().lower()
            # ignore MWEs
            if len(tokenizer.tokenize(term)) > 1: 
                continue
            sr2terms[row['subreddit']].append(term)
    return sr2terms

def get_sr2terms_original(): 
    sr2terms = defaultdict(list)
    with open(TERMS, 'r') as infile: 
        reader = csv.DictReader(infile, delimiter=',')
        for row in reader:
            term = row['term'].strip().lower()
            sr2terms[row['subreddit']].append(term)
    return sr2terms

def total_recall():
    '''
    See how many glossary words have scores computed for them
    ''' 
    sr2terms = get_sr2terms_no_mwes()
    # total possible recall of glossary words
    total_count = 0
    recall_count = 0
    for filename in sorted(os.listdir(PMI_PATH)): 
        subreddit = filename.replace('.csv', '').replace('_0.2', '')
        if subreddit not in sr2terms: continue
        gloss_terms = set(sr2terms[subreddit])
        scored_words = set()
        with open(PMI_PATH + filename, 'r') as infile: 
            reader = csv.DictReader(infile)
            for row in reader: 
                w = row['word'] 
                score = float(row['pmi'])
                if w in gloss_terms: 
                    recall_count += 1
                scored_words.add(w)
        print(subreddit, gloss_terms - scored_words)
        total_count += len(gloss_terms)
    print(recall_count / total_count)

def get_sr2terms():
    '''
    Only load glossary words that appear in the dataset 
    ''' 
    with open(ROOT + 'logs/existing_gloss_terms.json', 'r') as infile: 
        exist_gloss_terms = json.load(infile)
    return exist_gloss_terms 
        
def sense_vocab_coverage(): 
    sr2terms = get_sr2terms()
    sense_vocab_path = ROOT + 'logs/sr_sense_vocab/'
    big_vocab = set()
    with open(ROOT + 'logs/vocabs/10_1_filtered', 'r') as infile: 
        for line in infile: 
            big_vocab.add(line.strip())
    coverage = [] 
    freq_coverage = [] 
    for sr in sr2terms: 
        inpath = sense_vocab_path + sr + '_10.0' 
        vocab = set()
        with open(inpath, 'r') as infile: 
            for line in infile: 
                vocab.add(line.strip())
        terms = set(sr2terms[sr])
        top_percent = vocab & terms
        common = big_vocab & top_percent
        print(sr, len(common), common)
        freq_coverage.append(len(top_percent))
        coverage.append(len(common)) 
    print("AVERAGE # OF TERMS IN TOP PERCENT", np.mean(freq_coverage))
    print("AVERAGE # OF TERMS IN VOCAB:", np.mean(coverage))
    
def count_exact_string_matches(): 
    """
    Get glossary terms that actually
    show up in their subreddit based on exact string matches
    """
    sr2terms = get_sr2terms_no_mwes()
    # count it for each subreddit
    existing_terms = defaultdict(set) # sr : set of terms that show up
    num_mwe = 0 
    for sr in sr2terms:
        print(sr)
        terms = sr2terms[sr] 
        with open(ROOT + 'subreddits_month/' + sr + '/RC_sample', 'r') as infile: 
            for line in infile: 
                if not line.startswith('USER1USER0USER'): 
                    for term in terms: 
                        res = re.findall(r'\b' + re.escape(term) + r'\b', line.lower())
                        if len(res) > 0: 
                            existing_terms[sr].add(term)
        print(len(existing_terms[sr]))
    out_d = {}
    for sr in existing_terms: 
        out_d[sr] = list(existing_terms[sr])
    with open(ROOT + 'logs/existing_gloss_terms.json', 'w') as outfile: 
        json.dump(out_d, outfile)
        
def compare_gloss_dicts(): 
    with open(ROOT + 'logs/existing_gloss_terms.json', 'r') as infile: 
        exist_gloss_terms = json.load(infile)
    sr2terms = get_sr2terms_original()
    existing_count = 0
    all_count = 0
    unique_words = set()
    term2srs = defaultdict(list)
    for sr in sr2terms: 
        exist = set(exist_gloss_terms[sr])
        for w in exist: 
            term2srs[w].append(sr)
        unique_words.update(exist)
        existing_count += len(exist)
        all_t = set(sr2terms[sr])
        all_count += len(all_t)
        print(sr, len(all_t - exist), all_t - exist)
    shared_w = 0
    for w in term2srs: 
        if len(term2srs[w]) > 1: 
            shared_w += 1
            print(w, term2srs[w])
    print("Number of words shared across glossaries:", shared_w)
    print("Number of unique words in our evaluation:", len(unique_words))
    print("Number of words in our evaluation:", existing_count)
    print("Total number of glossary words (including MWEs):", all_count)
    
def sanity_check_gloss_words(): 
    '''
    Check that the glossary words 
    appear on the subreddit glossary html pages
    '''
    sr2terms = get_sr2terms_original()
    for sr in sr2terms: 
        if sr == 'boxoffice': 
            with open(WEBPAGES + sr + '1', 'r') as infile: 
                contents = infile.read().lower()
            with open(WEBPAGES + sr + '2', 'r') as infile: 
                contents += '\n' + infile.read().lower()
        elif sr == 'me_irl': # typo, was corrected later
            with open(WEBPAGES + 'mechanicalkeyboards', 'r') as infile: 
                contents = infile.read().lower()
        else:
            with open(WEBPAGES + sr, 'r') as infile: 
                contents = infile.read().lower()
        for term in sr2terms[sr]: 
            if term not in contents: 
                print(sr, term)

def main(): 
    basic_stats()
    #sense_vocab_coverage()
    #count_exact_string_matches()
    #compare_gloss_dicts()
    #sanity_check_gloss_words()

if __name__ == '__main__':
    main()
