"""
MWE discovery 
1. Run Stanford CoreNLP POS tagger over data  
2. Find possible MWE
3. Use a frequency and PMI threshold to filter out possible MWE 
4. Process the input data to concatenate MWE

There is an incompatibility with phrasemachine and transformers
so when using one we can't use the other. 
"""
from collections import Counter, defaultdict
import json
import os
import math
import csv
import numpy as np
from transformers import BasicTokenizer
#import phrasemachine
import time
import random 
import re

ROOT = '/mnt/data0/lucy/ingroup_lang/'
POS_INPUT = ROOT + 'subreddits_pos/'
LOGS = ROOT + 'logs/'
MWE_COUNTS = LOGS + 'mwe_counts/'
PMI = LOGS + 'pmi/' 
TERMS = ROOT + 'data/glossaries.csv'
SENSE_VOCAB = LOGS + 'vocabs/10_1_filtered'

def collect_patterns_counts2(subreddit): 
    """
    Justeson & Katz
    """
    start = time.time()
    nouns = set(['NN', 'NNS', 'NNP', 'NNPS'])
    adj = set(['JJ', 'JJR', 'JJS'])
    prep = 'IN' 
    input_file = POS_INPUT + subreddit + '/RC_sample.conll'
    prev_prev_tag = None
    prev_prev_word = None
    prev_tag = None
    prev_word = None
    mwe_counts = Counter()
    word_counts = Counter()
    with open(input_file, 'r') as infile: 
        for line in infile: 
            if line.strip() == '' or line.startswith('USER1USER0USER'): 
                curr_word = ''
                curr_tag = ''
            else: 
                items = line.strip().split('\t')
                curr_word = items[0].lower()
                curr_tag = items[1]
                word_counts[curr_word] += 1
            # A N
            # N N
            if (prev_tag in nouns or prev_tag in adj) and curr_tag in nouns: 
                mwe_counts[prev_word + ' ' + curr_word] += 1
            # A A N
            # A N N 
            # N A N 
            # N N N 
            if (prev_prev_tag in nouns or prev_prev_tag in adj) and \
                (prev_tag in nouns or prev_tag in adj) and curr_tag in nouns:
                mwe_counts[prev_prev_word + ' ' + prev_word + ' ' + curr_word] += 1
            # N P N 
            if prev_prev_tag in nouns and prev_tag == prep and curr_tag in nouns: 
                mwe_counts[prev_prev_word + ' ' + prev_word + ' ' + curr_word] += 1
            prev_prev_tag = prev_tag
            prev_tag = curr_tag
            prev_prev_word = prev_word
            prev_word = curr_word
    print("TIME:", time.time() - start)
    with open(MWE_COUNTS + subreddit + '.json', 'w') as outfile: 
        json.dump(mwe_counts, outfile)
    with open(MWE_COUNTS + subreddit + '_indivword.json', 'w') as outfile: 
        json.dump(word_counts, outfile)

def collect_patterns_counts(subreddit): 
    """
    Phrasemachine
    @input: 
	- subreddit name
    @output: 
        - subreddit_pm.json: a dictionary from multiword expression to its freq 
	- subreddit_indivword.json: a dictionary from word to its freq
    """
    start = time.time()
    input_file = POS_INPUT + subreddit + '/RC_sample.conll'
    mwe_counts = Counter()
    word_counts = Counter()
    with open(input_file, 'r') as infile: 
        curr_sent = []
        curr_pos = []
        i = 0
        for line in infile: 
            if line.strip() == '' or line.startswith('USER1USER0USER'): 
                if i == 20000: # process in this many sentence chunks
                    if len(curr_sent) > 1: 
                        res = phrasemachine.get_phrases(tokens=curr_sent, postags=curr_pos)
                        mwe_counts = mwe_counts + res['counts']
                    curr_sent = []
                    curr_pos = []
                    i = 0
                i += 1
                curr_sent.append('')
                curr_pos.append('')
            else: 
                items = line.strip().split('\t')
                curr_word = items[0].lower()
                curr_tag = items[1]
                word_counts[curr_word] += 1
                curr_sent.append(curr_word)
                curr_pos.append(curr_tag)
        res = phrasemachine.get_phrases(tokens=curr_sent, postags=curr_pos) # fence post
        mwe_counts = mwe_counts + res['counts']
    print("TIME:", time.time() - start)
    with open(MWE_COUNTS + subreddit + '_pm.json', 'w') as outfile: 
        json.dump(mwe_counts, outfile)
    with open(MWE_COUNTS + subreddit + '_indivword.json', 'w') as outfile: 
        json.dump(word_counts, outfile) 

def get_probable_mwe_all(freq_cutoff, pm=True): 
    """
    Calculates NPMI based on frequencies across all subreddits
    only if the MWE occurs frequently enough.
    This is implemented as a single function instead of split into
    NPMI calculation and a frequency cutoff functions
    to avoid iterating over all subreddits too many times. 
    @input: 
    - subreddit_pm.json (MWE counts)
    - subreddit_indivword.json (word counts)
    @output: 
    - all_pm_pmi.json (MWE NPMI scores if the MWE occurs more often than freq_cutoff)
    """
    pmi_path = MWE_COUNTS + 'all_pmi.json'
    if pm: pmi_path = MWE_COUNTS + 'all_pm_pmi.json'
    total_mwe_counts = Counter()
    total_word_counts = Counter()
    for subreddit in os.listdir(POS_INPUT): 
        print(subreddit)
        count_path = MWE_COUNTS + subreddit + '.json'
        if pm: 
            count_path = MWE_COUNTS + subreddit + '_pm.json'
        with open(count_path, 'r') as infile: 
            mwe_counts = Counter(json.load(infile))
            total_mwe_counts = total_mwe_counts + mwe_counts
        with open(MWE_COUNTS + subreddit + '_indivword.json', 'r') as infile: 
            word_counts = Counter(json.load(infile))
            total_word_counts = total_word_counts + word_counts
    N = float(sum(list(total_word_counts.values())))
    mwe_pmi = Counter()
    for mwe in total_mwe_counts: 
        if total_mwe_counts[mwe] < freq_cutoff: continue
        words = mwe.split(' ')
        indiv_prod = 1
        for w in words: 
            indiv_prod *= total_word_counts[w]
        mi = math.log(total_mwe_counts[mwe]*N/indiv_prod)
        mwe_pmi[mwe] = mi / (-math.log(total_mwe_counts[mwe]/N))
    with open(pmi_path, 'w') as outfile: 
        json.dump(mwe_pmi, outfile)

def get_probable_mwe(subreddit, pm=True): 
    """
    Calculates NPMI based on frequencies in each subreddit. 
    @inputs
	- pm: whether we are using justeson & katz or phrasemachine (pm)
        - subreddit_pm.json: MWE counts
        - subreddit_indivword.json: individual word counts
    @outputs: 
        - a mapping from MWE to NPMI score 
    """
    count_path = MWE_COUNTS + subreddit + '.json'
    pmi_path = MWE_COUNTS + subreddit + '_pmi.json'
    if pm: 
        count_path = MWE_COUNTS + subreddit + '_pm.json'
        pmi_path = MWE_COUNTS + subreddit + '_pm_pmi.json'
    with open(count_path, 'r') as infile: 
        mwe_counts = Counter(json.load(infile))
    with open(MWE_COUNTS + subreddit + '_indivword.json', 'r') as infile: 
        word_counts = Counter(json.load(infile))
    N = float(sum(list(word_counts.values())))
    mwe_pmi = Counter()
    for mwe in mwe_counts: 
        words = mwe.split(' ')
        indiv_prod = 1
        for w in words: 
            indiv_prod *= word_counts[w]
        mi = math.log(mwe_counts[mwe]*N/indiv_prod)
        mwe_pmi[mwe] = mi / (-math.log(mwe_counts[mwe]/N))
    with open(pmi_path, 'w') as outfile: 
        json.dump(mwe_pmi, outfile)

def get_high_freq_prob_mwe(subreddit, freq_cutoff, pm=True):
    '''
    Get only the MWE and their NPMI scores if the MWE is above some frequency threshold. 
    '''
    count_path = MWE_COUNTS + subreddit + '.json'
    pmi_path = MWE_COUNTS + subreddit + '_pmi.json'
    freq_path = MWE_COUNTS + subreddit + '_freq_pmi.json'
    if pm: 
        count_path = MWE_COUNTS + subreddit + '_pm.json'
        pmi_path = MWE_COUNTS + subreddit + '_pm_pmi.json'
        freq_path = MWE_COUNTS + subreddit + '_pm_freq_pmi.json'
    with open(pmi_path, 'r') as infile: 
        mwe_pmi = Counter(json.load(infile))
    with open(count_path, 'r') as infile: 
        mwe_counts = Counter(json.load(infile))
    mwe_freq = Counter()
    for mwe in mwe_pmi: 
        if mwe_counts[mwe] >= freq_cutoff: 
           mwe_freq[mwe] = mwe_pmi[mwe]
    with open(freq_path, 'w') as outfile:
        json.dump(mwe_freq, outfile)

def get_frequency_cutoff(): 
    """
    Get the frequency cutoff for top 20% of most freq words in each subreddit
    """
    min_cs = []
    for f in os.listdir(PMI): 
        min_c = float('inf')
        if not f.endswith('_0.2.csv'): continue
        with open(PMI + f, 'r') as infile: 
            reader = csv.DictReader(infile)
            for row in reader: 
                w = row['word']
                c = float(row['count'])
                if c < min_c: 
                    min_c = c
        min_cs.append(min_c) 
    print("Minimum count:", min(min_cs))
    print("Average minimum count:", np.mean(min_cs))

def get_glossary_pmi(pmi_cutoff, pm=True, indiv=False): 
    """
    Understand coverage of MWE glossaries with different NPMI and frequency cutoffs. 
    @inputs:
    - pmi_cutoff: cutoff for NPMI
    - pm: whether we are using phrasemachine
    - indiv: whether we are using NPMI scores based on individual subreddits
    or for all of reddit 
    """
    suffix = '_freq_pmi.json'
    all_red = 'all_pmi.json'
    if pm:
        suffix = '_pm_freq_pmi.json'
        all_red = 'all_pm_pmi.json'
    tokenizer = BasicTokenizer(do_lower_case=True)
    sr2terms = defaultdict(list)
    with open(TERMS, 'r') as infile: 
        reader = csv.DictReader(infile, delimiter=',')
        for row in reader:
            term = row['term'].strip().lower()
            if len(tokenizer.tokenize(term)) > 1: 
                sr2terms[row['subreddit']].append(term)
    missing_counts = []
    pmi_vals = []
    counted = 0
    with open(MWE_COUNTS + all_red, 'r') as infile:
        mwe_freq = Counter(json.load(infile))
    print("NUMBER OF MWES:", len(mwe_freq))
    for sr in sr2terms: 
        missing_count = 0
        if indiv: 
            with open(MWE_COUNTS + sr + suffix, 'r') as infile:
                mwe_freq = Counter(json.load(infile))
        for term in sr2terms[sr]: 
            if term in mwe_freq: 
                pmi_vals.append(mwe_freq[term])
                if mwe_freq[term] >= pmi_cutoff: 
                    counted += 1
                else: 
                    missing_count += 1
            else: 
                missing_count += 1
        missing_counts.append(missing_count)
    print("Number of MWEs in glossaries being considered:", counted)
    print("Average number of missing MWE per subreddit:", np.mean(missing_counts))
    print("Number of MWEs that aren't detected:", sum(missing_counts))
    print("Average PMI of glossary MWEs:", np.mean(pmi_vals))

def examine_words_around_cutoff(npmi_cutoff, pm=True, indiv=False): 
    mwes = []
    suffix = '_freq_pmi.json'
    all_red = 'all_pmi.json'
    if pm: 
        suffix = '_pm_freq_pmi.json'
        all_red = 'all_pm_pmi.json'
    if indiv: 
        for sr in os.listdir(POS_INPUT): 
           with open(MWE_COUNTS + sr + suffix, 'r') as infile: 
               mwe_freq = Counter(json.load(infile))
           for term in mwe_freq: 
               if mwe_freq[term] >= npmi_cutoff and mwe_freq[term] <= npmi_cutoff + 0.01: 
                   mwes.append(term)
    else: 
       with open(MWE_COUNTS + all_red, 'r') as infile:
           mwe_freq = Counter(json.load(infile))
       for term in mwe_freq: 
           if mwe_freq[term] >= npmi_cutoff and mwe_freq[term] <= npmi_cutoff + 0.01: 
               mwes.append(term)
    random.shuffle(mwes)
    print(mwes[:100])

def count_exact_string_matches(): 
    """
    Number of multi-word glossary terms that actually
    show up in their subreddit based on exact string matches
    """
    tokenizer = BasicTokenizer(do_lower_case=True)
    sr2terms = defaultdict(list)
    with open(TERMS, 'r') as infile: 
        reader = csv.DictReader(infile, delimiter=',')
        for row in reader:
            term = row['term'].strip().lower()
            if len(tokenizer.tokenize(term)) > 1: 
                sr2terms[row['subreddit']].append(term)
    # count it for each subreddit
    counts = defaultdict(Counter)
    num_mwe = 0 
    for sr in sr2terms:
        print(sr)
        mwes = sr2terms[sr] 
        with open(ROOT + 'subreddits_month/' + sr + '/RC_sample', 'r') as infile: 
            for line in infile: 
                if not line.startswith('USER1USER0USER'): 
                    for mwe in mwes: 
                        res = re.findall(r'\b' + re.escape(mwe) + r'\b', line.lower())
                        if len(res) > 0: 
                            counts[sr][mwe] += len(res) 
        for mwe in counts[sr]: 
            if counts[sr][mwe] > 0: 
                num_mwe += 1
    print("# of glossary MWEs that show up in their subreddits:", num_mwe)
    with open(ROOT + 'logs/mwe_direct_matches.json', 'w') as outfile: 
        json.dump(counts, outfile)

def generate_mwe_storage_dict(npmi_cutoff): 
    npmi_path = MWE_COUNTS + 'all_pm_pmi.json'
    outpath = LOGS + 'all_mwes.json'
    res = {}
    with open(npmi_path, 'r') as infile:
        mwe_pmi = json.load(infile)
    num_mwe = 0
    for mwe in mwe_pmi: 
        if mwe_pmi[mwe] >= npmi_cutoff: 
            parts = mwe.split(' ')
            if len(parts) == 0: continue
            num_mwe += 1
            curr_layer = res
            for w in parts: 
                if w not in curr_layer: 
                    curr_layer[w] = {}
                curr_layer = curr_layer[w]
            curr_layer['$END_TOKEN$'] = {}
    print("# of MWE:", num_mwe)
    with open(outpath, 'w') as outfile:
        json.dump(res, outfile) 
        
def get_vocab_mwes(): 
    count = 0
    with open(SENSE_VOCAB, 'r') as infile:
        for line in infile: 
            l = line.strip()
            if len(l.split(' ')) > 1: 
                count += 1
                print(l)
    print(count)
                
def main():
    random.seed(0)
    #for subreddit in os.listdir(POS_INPUT): 
    #    print(subreddit)
    #    collect_patterns_counts2(subreddit)
    #    get_probable_mwe(subreddit, pm=False)
    #    get_high_freq_prob_mwe(subreddit, 2, pm=False)
    #count_exact_string_matches()
    #get_probable_mwe_all(2, pm=False)
    #get_probable_mwe_all(2)
    #get_glossary_pmi(-float('inf'), pm=False)
    #examine_words_around_cutoff(0.10)
    #count_exact_string_matches()
    #get_frequency_cutoff()
    #generate_mwe_storage_dict(0.20)
    get_vocab_mwes()

if __name__ == '__main__': 
    main()
