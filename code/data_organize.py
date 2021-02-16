"""
Functions for organizing or reformatting 
Reddit data.

Examples of things covered in this script: 
- Counting the number of comments
- Creating separate documents for each subreddit
- Formatting training/test data for WSI, such as 
   sampling 500 examples of each vocab word for sense clustering

Python 2.7, since this was an early document in this project. 

though if you use stanfordnlp you should
use Python 3. 
"""
import json
import time
import re
import string
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SQLContext
from collections import Counter
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from functools import partial
from nltk.stem import WordNetLemmatizer
from transformers import BasicTokenizer, BertTokenizer
from glossary_eval import get_sr2terms
import numpy as np

wnl = WordNetLemmatizer()

#ROOT = '/data0/lucy/ingroup_lang/'
ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
#ROOT = '/mnt/data0/lucy/ingroup_lang/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
SR_FOLDER_MONTH = ROOT + 'subreddits_month/'
SR_FOLDER = ROOT + 'subreddits/'
SR_FOLDER2 = ROOT + 'subreddits2/'
SUBREDDITS = DATA + 'subreddit_list.txt'
REMOVED_SRS = DATA + 'non_english_sr.txt'
AMRAMI_INPUT = '/global/scratch/lucy3_li/bertwsi/reddit_input/'

conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
reddits = set()

def clean_up_text(text): 
    # remove urls
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                  '', text, flags=re.MULTILINE)
    # remove new lines
    text = text.replace('\n', ' ').replace('\r', ' ')
    # remove removed comments
    if text.strip() == '[deleted]' or text.strip() == '[removed]': 
        text = ''
    # standardize usernames and subreddit names
    text = re.sub('u/[A-Za-z_0-9-]+', 'u/USER', text)
    text = re.sub('r/[A-Za-z_0-9]+', 'r/SUBREDDIT', text)
    # replace numbers except when they occur with alphabetic characters
    text = re.sub('(?<![A-Za-z0-9])(\d+)(?![A-Za-z0-9])', '<num0-9>', text) 
    return text

def get_comment(line): 
    """
    Called by create_subreddit_docs() 
    @input: 
         - a dictionary containing a comment
    @output: 
         - a tuple (subreddit name: comment text)
    
    Usernames start with u/, can have underscores, dashes, alphanumeric letters.
    Subreddits start with r/, can have underscores and alphanumberic letters. 
    Example of how the number regex works: 
    a = 'his table34 is -3492, -998, and 3.0.4 and 08:30 and 23-389! calcul8 this ple3se'
    a = 'his table34 is -NUM, -NUM, and NUM.NUM.NUM and NUM:NUM and NUM-NUM! calcul8 this ple3se' 
    """
    comment = json.loads(line)
    text = clean_up_text(comment['body']) 
    return (comment['subreddit'].lower(), text)

def get_comment_user(line): 
    """
    Called by create_subreddit_docs()
    @input: 
         - a dictionary containing a comment
    @output: 
         - a tuple (subreddit name: comment text)
    
    same as get_comment(line) except the key is both subreddit and user
    """
    comment = json.loads(line)
    text = clean_up_text(comment['body']) 
    return ((comment['subreddit'].lower(), comment['author'].lower()), text)
    
def subreddit_of_interest(line): 
    '''
    Returns only subreddits in the set of reddits we want. 
    '''
    comment = json.loads(line)
    return 'subreddit' in comment and 'body' in comment and \
        comment['subreddit'].lower() in reddits
            
def get_subreddit(line): 
    '''
    Returns (subreddit, 1)
    '''
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment and \
        comment['body'].strip() != '[deleted]' and comment['body'].strip() != '[removed]': 
        return (comment['subreddit'].lower(), 1)
    else: 
        return (None, 0)

def get_subreddit_json(line): 
    '''
    Each comment is a json, but some jsons contain unwanted deleted comments. 
    
    Returns: (subreddit, json)
    '''
    comment = json.loads(line) 
    if 'subreddit' in comment and 'body' in comment and \
        comment['body'].strip() != '[deleted]' and comment['body'].strip() != '[removed]': 
        return (comment['subreddit'].lower(), [line])
    else: 
        return (None, [])

def count_comments_for_one_subreddit(sr): 
    '''
    Output: Number of comments in subreddit sr
    '''
    path = DATA + 'RC_all'
    data = sc.textFile(path)
    data = data.filter(lambda line: json.loads(line)['subreddit'].lower() == sr) 
    print("NUMBER OF COMMENTS IN " + sr.upper() + " IS " + str(data.count()))

def get_top_subreddits(n=300): 
    '''
    Get the top n subreddits by number of comments. 
    Takes ~30 min for 1 month on redwood using --master 'local[*]'
    '''
    path = DATA + 'RC_all'
    data = sc.textFile(path)
    data = data.map(get_subreddit)
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    data = data.collectAsMap()
    sr_counts = Counter(data)
    with open(SUBREDDITS, 'w') as outfile: 
        for sr in sr_counts.most_common(n): 
            outfile.write(sr[0] + '\n') 

def sample_lines(tup): 
    '''
    Called by create_subreddit_docs()
    
    Each subreddit has 80k comments (each line is a comment)
    '''
    sr = tup[0]
    lines = tup[1]
    assert len(lines) >= 80000,"OH NO THE SUBREDDIT " + sr + \
        " IS TOO SMALL AND HAS ONLY " + str(len(lines)) + " LINES." 
    new_lines = random.sample(lines, 80000)
    return new_lines
            
def save_doc(item): 
    '''
    Save each item (subreddit) as a document. 
    '''
    if item[0] is not None:
        path = SR_FOLDER_MONTH + item[0] + '/'
        with open(path + 'RC_sample', 'w') as file:
            file.write(item[1].encode('utf-8', 'replace'))
    
def create_subreddit_docs(): 
    '''
    Create a document for each subreddit by month
    Lines that start with USER1USER0USER are usernames
    whose comments on that subreddit then follow.

    The step after this is to move non-English subreddits
    from this folder so they are not part of the remainder of 
    the pipeline. 
    '''
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            sr = line.strip().lower()
            reddits.add(sr)
    # create output folders
    for sr in reddits: 
        path = SR_FOLDER_MONTH + sr + '/'
        if not os.path.exists(path): 
            os.makedirs(path)
    random.seed(0)
    logfile = open(LOGS + 'create_subreddit_docs.temp', 'w') 
    
    path = DATA + 'RC_all'
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    data = data.map(get_subreddit_json)  
    data = data.reduceByKey(lambda n1, n2: n1 + n2) 
    data = data.filter(lambda tup: tup[0] is not None)
    data = data.flatMap(sample_lines) 
    logfile.write('After flatmap: ' + str(data.count()) + '\n') 
    data = data.map(get_comment_user)
    logfile.write("After mapping to comment user: " + str(data.count()) + '\n') 
    data = data.reduceByKey(lambda n1, n2: n1 + '\n' + n2)
    logfile.write("Number of subreddit-user pairs: " + str(data.count()) + '\n')
    data = data.map(lambda tup: (tup[0][0], 'USER1USER0USER' + str(''.join(format(ord(i), 'b') for i in tup[0][1])) + '\n' + tup[1]))
    data = data.reduceByKey(lambda n1, n2: n1 + '\n' + n2)
    data = data.foreach(save_doc)
    logfile.close()

def count_comments_all(): 
    '''
    This counts the number of comments per subreddit. 
    '''
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            sr = line.strip().lower()
            reddits.add(sr)
    random.seed(0)
    path = DATA + "RC_all"
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    data = data.map(get_subreddit)  
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    c = Counter(data.collectAsMap())
    outfile = open(LOGS + "sr_comment_counts", 'w')
    for sr in c.most_common():
        if sr[0] is None: continue 
        outfile.write(sr[0] + '\t' + str(sr[1]) + '\n') 
    outfile.close()

def sentences_with_target_words(line, tokenizer=None, target_set=set()): 
    '''
    Used by filter_ukwac() to find sentences with target words in ukwac. 
    '''
    sentences = sent_tokenize(line.strip()) 
    ret = []
    for s in sentences: 
        tokens = set(tokenizer.tokenize(s.lower()))
        for item in target_set: 
            lemma = item.split('.')[0]
            pos = item.split('.')[1]
            if pos == 'j': pos = 'a' # adjective
            for tok in tokens: 
                if wnl.lemmatize(tok, pos) == lemma: 
                    ret.append((lemma + '.' + pos, tok, s))
    return ret

def count_ukwac_lemmas(tup): 
    """
    This checks that we have enough examples
    of each lemma in our dataset for sense induction. 
    We hope to have >500 examples. 
    """
    lemma = tup[0]
    return (lemma, 1)

def filter_ukwac(): 
    """
    For every sentence, only keep the ones in which a targeted lemma appears. 
    Format it as 
    lemma \t target word \t sentence
    for easy BERT processing :)
    
    ukwac is the training data for the SemEval 2013 WSI task. 
    """
    directory = ROOT + 'SemEval-2013-Task-13-test-data/contexts/xml-format/'
    target_set = set()
    for f in os.listdir(directory): 
        if f.endswith('.xml'): 
            target_set.add(f.replace('.xml', ''))
    data = sc.textFile(DATA + 'ukwac_preproc')
    data = data.filter(lambda line: not line.startswith("CURRENT URL "))
    tokenizer = BasicTokenizer(do_lower_case=True) 
    data = data.flatMap(partial(sentences_with_target_words, tokenizer=tokenizer, 
        target_set=target_set))
    data = data.sample(False,0.05,0)
    counts = data.map(count_ukwac_lemmas) 
    counts = counts.reduceByKey(lambda n1, n2: n1 + n2)
    counts = counts.collectAsMap()
    data = data.collect()
    with open(LOGS + 'ukwac2.txt', 'w') as outfile: 
        for item in data: 
            outfile.write(item[0] + '\t' + item[1] + '\t' + item[2] + '\n')
    for lemma in counts: 
        print(lemma, counts[lemma])

def prep_finetuning_old(): 
    """
    Remove usernames, concatenate files together
    This was for use with Huggingface's run_lm_finetuning.py. 
    
    Our paper does not include the domain adapted model. 
    """
    rdds = []
    for folder in os.listdir(SR_FOLDER_MONTH): 
        path = SR_FOLDER_MONTH + folder + '/RC_sample'
        data = sc.textFile(path)
        data = data.filter(lambda line: not line.startswith('USER1USER0USER'))
        rdds.append(data)
    all_data = sc.union(rdds)
    test, train = all_data.randomSplit(weights=[0.1, 0.9], seed=1)
    test = sc.parallelize(test.takeSample(False, 1000000, 0))
    train = sc.parallelize(train.takeSample(False, 10000000, 0))
    test.coalesce(1).saveAsTextFile(LOGS + 'finetune_input_test2')
    train.coalesce(1).saveAsTextFile(LOGS + 'finetune_input_train2')
    
def prep_finetuning_part1(): 
    """
    Output: a parquet where each cell is a comment 
    
    Our paper does not include the domain adapted model. 
    """
    rdds = []
    for folder in os.listdir(SR_FOLDER_MONTH): 
        path = SR_FOLDER_MONTH + folder + '/RC_sample'
        data = sc.textFile(path)
        data = data.filter(lambda line: not line.startswith('USER1USER0USER') and line.strip() != "")
        rdds.append(data)
    all_data = sc.union(rdds)
    test, train = all_data.randomSplit(weights=[0.1, 0.9], seed=1)
    test = test.map(lambda line: Row(comment=line))
    test_df = sqlContext.createDataFrame(test)
    test_df.write.mode('overwrite').parquet(LOGS + "finetune_input_test3/test.parquet")
    train = train.map(lambda line: Row(comment=line))
    train_df = sqlContext.createDataFrame(train)
    train_df.write.mode('overwrite').parquet(LOGS + "finetune_input_train3/train.parquet")
    
def prep_finetuning_part2():
    """
    Input: a parquet where each cell is a list of sentences and
    each cell is a single comment (or document)
    Output: a document where one sentence per line, 
    with documents separate by an additional newline.
    
    Our paper does not include the domain adapted model. 
    """
    test_df = sqlContext.read.parquet(LOGS + "finetune_input_test3/ssplit_test.parquet")
    test_num_sent = sum(test_df.rdd.map(lambda row: len(row.sen)).collect())
    #test_num_tokens = sum(test_df.rdd.map(lambda row: row.num_tokens).collect())
    test = test_df.rdd.map(lambda row: '\n'.join(row.sen) + '\n')
    test.saveAsTextFile(LOGS + 'finetune_input_test3/test')
    train_df = sqlContext.read.parquet(LOGS + "finetune_input_train3/ssplit_train.parquet")
    train_num_sent = sum(train_df.rdd.map(lambda row: len(row.sen)).collect())
    #train_num_tokens = sum(test_df.rdd.map(lambda row: row.num_tokens).collect())
    train = train_df.rdd.map(lambda row: '\n'.join(row.sen) + '\n')
    train.saveAsTextFile(LOGS + 'finetune_input_train3/train')
    print("---Number of sentences in test set:", test_num_sent)
    #print("---Number of tokens in test set:", test_num_tokens)
    print("---Number of sentences in train set:", train_num_sent)
    #print("---Number of tokens in train set:", train_num_tokens)

def est_finetuning_gloss_cov(): 
    """
    For words in subreddit glossaries, calculate how many times
    they appear in the finetuning input. 
    
    This was used during domain adaptation to check that glossary words
    are in fact being seen by the model. 
    
    Our paper does not include the domain adapted model. 
    """
    data = sc.textFile(LOGS + 'finetune_input_train2')
    sr2terms = get_sr2terms()
    terms = set()
    for sr in sr2terms: 
        terms.update(sr2terms[sr])
    tokenizer = BasicTokenizer(do_lower_case=True) 
    data = data.flatMap(lambda line: list(set(tokenizer.tokenize(line)) & terms))
    data = data.map(lambda w: (w, 1))
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    data = data.collectAsMap()
    gloss_examples = Counter(data)
    avg = sum(gloss_examples.values()) / len(gloss_examples)
    print("------RESULT median # of examples per gloss word:", np.median(list(gloss_examples.values())))
    print("------RESULT avg # of examples per gloss word:", avg)

def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def prep_finetuning2(num_epochs=3): 
    '''
    Take the finetuning input, shuffle it three times and output as chunks.
    This was used to domain adapt (finetune) BERT to Reddit data, but we found that
    this step did not improve results, so our paper does not include
    the domain adapted model. 
    '''
    filename = LOGS + 'finetune_input_train/part-00000'
    print("Reading in file...") 
    with open(filename, 'r') as infile: 
        lines = infile.readlines()
    for i in range(num_epochs): 
        print("Getting epoch", i)
        random.shuffle(lines)
        new_filename = filename + '_epoch' + str(i)
        j = 0
        print("Dividing chunks...") 
        for chunk in divide_chunks(lines, 10000000): 
            with open(new_filename + '_chunk' + str(j), 'w') as outfile: 
                outfile.writelines(chunk)
            j += 1

def get_vocab_word_instances(line, vocab=None):
    '''
    Used by sample_word_instances()
    to get a flatmap from each comment to (vocab word, [comment])
    '''
    tokenizer = BasicTokenizer(do_lower_case=True)
    line = line.strip()
    tokens = set(tokenizer.tokenize(line))
    ret = []
    union = tokens & set(vocab.keys())
    for w in union: 
        ret.append((w, [line]))
    if len(union) == 0: 
        ret.append((None, [line]))
    return ret 

def sample_vocab_lines(tup): 
    '''
    This function samples words 500 times. 
    It initially samples a larger number of words, 
    but then removes cases where the examples have many
    repetitions (such as comments written by bots). 
    
    Used by sample_word_instances()
    '''
    w = tup[0]
    lines = tup[1]
    sample_num = 25000
    if w == 'compose': # special case that occurs often
        sample_num = 300000
    instances = random.sample(lines, min(sample_num, len(lines)))
    tokenizer = BasicTokenizer(do_lower_case=True)
    comment2windows = {} # comment idx to window IDs
    windowIDs = {} # window list to window ID
    ID_counts = Counter() # window ID to count
    for i, inst in enumerate(instances): 
        lh, _, rh = inst.partition(w)
        ltokens = tokenizer.tokenize(lh)
        rtokens = tokenizer.tokenize(rh)
        ltokens = ltokens[-5:]
        rtokens = rtokens[:5]
        window = ltokens + [w] + rtokens
        if tuple(window) in windowIDs: 
            windowID = windowIDs[tuple(window)]
        else: 
            windowID = i 
            windowIDs[tuple(window)] = i
        comment2windows[i] = windowID 
        ID_counts[windowID] += 1
    new_instances = []
    for i, inst in enumerate(instances): 
        windowID = comment2windows[i]
        c = ID_counts[windowID]
        if c < 10: 
            new_instances.append(inst)
        if len(new_instances) == 500: break
    
    if len(new_instances) < 500 and len(lines) <= 20000: 
        print("Error: Not enough samples for word:", w)
        new_instances = []
    elif len(new_instances) < 500: 
        print("Error: Need to initially sample more comments for word:", w, len(lines), len(new_instances))
        new_instances = []
    return (w, new_instances)

def save_vocab_instances_doc(tup, vocab=None): 
    '''
    This is used by sample_word_instances()
    to write the output file. 
    '''
    w = tup[0]
    lines = tup[1]
    with open(LOGS + 'vocabs/docs/' + str(vocab[w]), 'w') as outfile:
        for l in lines: 
            outfile.write(l + '\n')

def sample_word_instances(): 
    '''
    This function takes in a vocab file of words and samples 500 instances. 
    '''
    vocab_file = LOGS + 'vocabs/10_1_filtered'
    vocab = {} 
    # map word to ID
    with open(vocab_file, 'r') as infile: 
        for i, line in enumerate(infile): 
            w = line.strip()
            vocab[w] = i
    with open(LOGS + 'vocabs/vocab_map.json', 'w') as outfile: 
        json.dump(vocab, outfile)
    
    rdds = []
    for folder in os.listdir(SR_FOLDER_MONTH): 
        path = SR_FOLDER_MONTH + folder + '/RC_sample'
        data = sc.textFile(path) 
        data = data.filter(lambda line: not line.startswith('USER1USER0USER'))
        data = data.flatMap(partial(get_vocab_word_instances, vocab=vocab))
        data = data.filter(lambda tup: tup[0] is not None)
        app = data.reduceByKey(lambda n1, n2: n1 + n2)
        rdds.append(app)
    all_tups = sc.union(rdds)
    all_tups = all_tups.reduceByKey(lambda n1, n2: n1 + n2)
    sample_tups = all_tups.map(sample_vocab_lines)
    sample_tups = sample_tups.foreach(partial(save_vocab_instances_doc, vocab=vocab))

def tokenizer_check(): 
    '''
    This function was used to compare two tokenizers. 
    
    The main conclusion after Lucy ran this function was that
    the BasicTokenizer is the BertTokenizer with the wordpieces
    connected together. 
    '''
    path = SR_FOLDER_MONTH + 'askreddit/RC_sample'
    data = sc.textFile(path)
    sample = data.takeSample(False, 100) 
    tokenizer1 = BasicTokenizer(do_lower_case=True)
    tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    success = True
    for s in sample: 
        tokens1 = tokenizer1.tokenize(s)
        tokens2 = tokenizer2.tokenize(s)
        prev_word = None
        tokens3 = []
        ongoing_word = []
        for w in tokens2: 
            if w.startswith('##'): 
                if not prev_word.startswith('##'): 
                    ongoing_word.append(prev_word)
                ongoing_word.append(w[2:])
            else: 
                if len(ongoing_word) == 0 and prev_word is not None: 
                    tokens3.append(prev_word)
                elif prev_word is not None:
                    tokens3.append(''.join(ongoing_word))
                ongoing_word = []
            prev_word = w
        if len(ongoing_word) == 0 and prev_word is not None: 
            tokens3.append(prev_word)
        elif prev_word is not None: 
            tokens3.append(''.join(ongoing_word))
        if tokens3 != tokens1: 
            print("OH NOOOOOOOOOO")
            print(tokens1)
            print(tokens3) 
            success = False
    if success: 
        print("TOKENS MATCHED UP!")

def get_all_examples_with_word(word): 
    '''
    This function gathers all examples that contain a word, 
    to use as input to Amrami & Goldberg's model. 
    '''
    rdds = []
    for folder in os.listdir(SR_FOLDER_MONTH): 
        path = SR_FOLDER_MONTH + folder + '/RC_sample'
        data = sc.textFile(path) 
        data = data.filter(lambda line: not line.startswith('USER1USER0USER'))
        tokenizer = BasicTokenizer(do_lower_case=True)
        data = data.filter(lambda line: word in set(tokenizer.tokenize(line.strip())))
        rdds.append(data)
    all_occ = sc.union(rdds)
    all_occ.coalesce(1).saveAsTextFile(AMRAMI_INPUT + word)
            
def main(): 
    #get_top_subreddits(n=500)
    #create_subreddit_docs()
    #create_sr_user_docs() 
    #filter_ukwac()
    #temp()
    #sample_word_instances()
    #est_finetuning_gloss_cov()
    sc.stop()

if __name__ == '__main__':
    main()
