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
#from stanfordnlp.server import CoreNLPClient

ROOT = '/data0/lucy/ingroup_lang/'
#ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
DATA = ROOT + 'data/'
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

def get_top_subreddits(n=300): 
    '''
    Get the top n subreddits by number
    of comments. 
    Takes ~30 min for 1 month on redwood
    using --master 'local[*]'
    '''
    path = DATA + 'RC_2019-05'
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.map(get_subreddit)
    data = data.reduceByKey(lambda n1, n2: n1 + n2)
    data = data.collectAsMap()
    sr_counts = Counter(data)
    with open(SUBREDDITS, 'w') as outfile: 
        for sr in sr_counts.most_common(n): 
            outfile.write(sr[0] + '\n') 
            
def save_doc(item): 
    if item[0] is not None:
        path = SR_FOLDER_MONTH + item[0] + '/'
        with open(path + MONTH, 'w') as file:
            file.write(item[1].encode('utf-8', 'replace'))
            
def save_user_doc(item): 
    sr = item[0][0]
    user = item[0][1]
    text = item[1]
    if sr is not None:
        path = SR_FOLDER + sr + '/'
        with open(path + MONTH + '_' + user, 'w') as file:
            file.write(text.encode('utf-8', 'replace'))
    
def create_subreddit_docs(): 
    '''
    Create a document for each subreddit by month
    '''
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
        path = SR_FOLDER_MONTH + sr + '/'
        if not os.path.exists(path): 
            os.makedirs(path)
    global MONTH
    MONTH = 'RC_2019-05'
    path = DATA + MONTH
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    data = data.filter(subreddit_of_interest)
    data = data.map(get_comment)
    data = data.reduceByKey(lambda n1, n2: n1 + '\n' + n2)
    data = data.foreach(save_doc)
    
def create_sr_user_docs(): 
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
    global MONTH
    MONTH = 'RC_2019-05'
    path = DATA + MONTH
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
    MONTH = 'RC_2019-05'
    global client
    client = CoreNLPClient(annotators=['tokenize','ssplit'], timeout=50000000, threads=18, memory='64G')
    for folder_name in os.listdir(SR_FOLDER): 
        if os.path.isdir(SR_FOLDER + folder_name):
            path = SR_FOLDER + folder_name + '/' + MONTH
            data = sc.textFile(path)
            data = data.map(process_comment)
            data.coalesce(1).saveAsTextFile(SR_FOLDER2 + folder_name)
            break

def main(): 
    #get_top_subreddits(n=500)
    #create_subreddit_docs()
    create_sr_user_docs() 
    sc.stop()

if __name__ == '__main__':
    main()
