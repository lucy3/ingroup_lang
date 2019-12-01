"""
Functions for organizing or reformatting 
Reddit data.

Python 2.7, 

though if you use stanfordnlp you should
use Python 3. 
"""
import json
import time
from tqdm import tqdm
import re
import string
import os
from pyspark import SparkConf, SparkContext
from collections import Counter
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from functools import partial
from nltk.stem import WordNetLemmatizer
#from stanfordnlp.server import CoreNLPClient

wnl = WordNetLemmatizer()

#ROOT = '/data0/lucy/ingroup_lang/'
ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
SR_FOLDER_MONTH = ROOT + 'subreddits_month/'
SR_FOLDER = ROOT + 'subreddits/'
SR_FOLDER2 = ROOT + 'subreddits2/'
SUBREDDITS = DATA + 'subreddit_list.txt'
REMOVED_SRS = DATA + 'non_english_sr.txt'

conf = SparkConf()
sc = SparkContext(conf=conf)
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
    Called by create_sr_user_docs() 
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
    comment = json.loads(line)
    return 'subreddit' in comment and 'body' in comment and \
        comment['subreddit'].lower() in reddits
            
def get_subreddit(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment and \
	    comment['body'].strip() != '[deleted]' and comment['body'].strip() != '[removed]': 
        return (comment['subreddit'].lower(), 1)
    else: 
        return (None, 0)

def get_subreddit_json(line): 
    comment = json.loads(line) 
    if 'subreddit' in comment and 'body' in comment and \
	    comment['body'].strip() != '[deleted]' and comment['body'].strip() != '[removed]': 
        return (comment['subreddit'].lower(), [line])
    else: 
        return (None, [])


def count_comments_for_one_subreddit(sr): 
    path = DATA + 'RC_all'
    data = sc.textFile(path)
    data = data.filter(lambda line: json.loads(line)['subreddit'].lower() == sr) 
    print("NUMBER OF COMMENTS IN " + sr.upper() + " IS " + str(data.count()))

def get_top_subreddits(n=300): 
    '''
    Get the top n subreddits by number
    of comments. 
    Takes ~30 min for 1 month on redwood
    using --master 'local[*]'
    '''
    path = DATA + 'RC_all'
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.map(get_subreddit)
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    data = data.collectAsMap()
    sr_counts = Counter(data)
    with open(SUBREDDITS, 'w') as outfile: 
        for sr in sr_counts.most_common(n): 
            outfile.write(sr[0] + '\n') 

def sample_lines(tup): 
    sr = tup[0]
    lines = tup[1]
    assert len(lines) >= 80000,"OH NO THE SUBREDDIT " + sr + \
    	" IS TOO SMALL AND HAS ONLY " + str(len(lines)) + " LINES." 
    new_lines = random.sample(lines, 80000)
    return new_lines
            
def save_doc(item): 
    if item[0] is not None:
        path = SR_FOLDER_MONTH + item[0] + '/'
        with open(path + 'RC_sample', 'w') as file:
            file.write(item[1].encode('utf-8', 'replace'))
            
def save_user_doc(item): 
    sr = item[0][0]
    user = item[0][1]
    text = item[1]
    if sr is not None:
        path = SR_FOLDER + sr + '/'
        with open(path + 'RC_sample_' + user, 'w') as file:
            file.write(text.encode('utf-8', 'replace'))
    
def create_subreddit_docs(): 
    '''
    Create a document for each subreddit by month
    Lines that start with @@#USER#@@_ are usernames
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
    #path = DATA + 'tinyData'
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
    
def create_sr_user_docs(): 
    """
    Creates one document per user per subreddit. 
    It's possible that this method requires too much runtime
    when it comes to I/O a gazillion tiny little docs while tokenizing. 
    """
    non_english_reddits = set()
    with open(REMOVED_SRS, 'r') as inputfile: 
        for line in inputfile: 
            non_english_reddits.add(line.strip().lower())
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            sr = line.strip().lower()
            if sr not in non_english_reddits: 
                reddits.add(sr)
    # create output folders
    for sr in reddits: 
        path = SR_FOLDER + sr + '/'
        if not os.path.exists(path): 
            os.makedirs(path)

    path = DATA + 'RC_sample'
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    data = data.map(get_comment_user)
    data = data.reduceByKey(lambda n1, n2: n1 + '\n' + n2)
    data = data.foreach(save_user_doc)
    
def process_comment(line): 
    if line.strip().lower() == '[deleted]' or \
        line.strip().lower() == '[removed]':
        return ''
    ann = client.annotate(line)
    new_line = ''
    for s in ann.sentence: 
        for t in s.token: 
            new_line += t.word.lower() + ' '
    return new_line
    
def tokenize_docs(): 
    """
    Lowercases and tokenizes documents.
    NOTE: Takes about as long as the non-Spark version so not being used. 
    """
    MONTH = 'RC_sample'
    global client
    client = CoreNLPClient(annotators=['tokenize','ssplit'], timeout=50000000, threads=18, memory='64G')
    for folder_name in os.listdir(SR_FOLDER): 
        if os.path.isdir(SR_FOLDER + folder_name):
            path = SR_FOLDER + folder_name + '/' + MONTH
            data = sc.textFile(path)
            data = data.map(process_comment)
            data.coalesce(1).saveAsTextFile(SR_FOLDER2 + folder_name)
            break

def sample_subreddits(): 
    """
    This function keeps getting killed
    on both savio and redwood. 
    """
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            sr = line.strip().lower()
            reddits.add(sr)
    random.seed(0)
    path = DATA + "RC_all"
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    data = data.map(get_subreddit_json)  
    data = data.reduceByKey(lambda n1, n2: n1 + n2) 
    data = data.filter(lambda tup: tup[0] is not None)
    data = data.map(sample_lines) 
    sr_lines = data.collectAsMap()
    outpath = DATA + "RC_sample"
    outfile = open(outpath, 'w') 
    for sr in sr_lines: 
        if sr is None: continue
        print("WRITING " + sr.upper() + " TO FILE") 
        lines = sr_lines[sr]
        print("NUMBER OF LINES:", len(lines))
        for l in lines: 
            outfile.write(l + '\n') 
    outfile.close()

def count_comments_all(): 
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

def sentences_with_target_words(line, target_set=set()): 
    sentences = sent_tokenize(line.strip()) 
    ret = []
    for s in sentences: 
       tokens = set(word_tokenize(s.lower()))
       for item in target_set: 
           lemma = item.split('.')[0]
           for tok in tokens: 
               if wnl.lemmatize(tok) == lemma: 
                   ret.append((lemma, tok, s))
    return ret

def filter_ukwac(): 
    """
    For every sentence
    Only keep the ones in which a targeted lemma
    appears. 
    Format it as 
    lemma \t target word \t sentence
    for easy BERT processing :)
    """
    directory = ROOT + 'SemEval-2013-Task-13-test-data/contexts/xml-format/'
    target_set = set()
    for f in os.listdir(directory): 
        if f.endswith('.xml'): 
            target_set.add(f.replace('.xml', ''))
    data = sc.textFile(DATA + 'ukwac_preproc.gz')
    data = data.filter(lambda line: not line.startswith("CURRENT URL "))
    data = data.flatMap(partial(sentences_with_target_words, target_set=target_set))
    data = data.collect()
    with open(LOGS + 'ukwac2.txt', 'w') as outfile: 
        for item in data: 
            outfile.write(item[0] + '\t' + item[1] + '\t' + item[2] + '\n')

def prep_finetuning(): 
    """
    Remove usernames, concatenate files together
    """
    rdds = []
    for folder in os.listdir(SR_FOLDER_MONTH): 
        path = SR_FOLDER_MONTH + folder + '/RC_sample'
        data = sc.textFile(path)
        data = data.filter(lambda line: not line.startswith('USER1USER0USER'))
        rdds.append(data)
    all_data = sc.union(rdds)
    all_data.saveAsTextFile(LOGS + 'finetune_input') 
    
def main(): 
    #get_top_subreddits(n=500)
    #create_subreddit_docs()
    #create_sr_user_docs() 
    #prep_finetuning()
    filter_ukwac()
    sc.stop()

if __name__ == '__main__':
    main()
