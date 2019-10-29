'''
Create networks for each subreddit
'''
import sys
from collections import Counter
import json
from pyspark import SparkConf, SparkContext
from functools import partial

ROOT = '/data0/lucy/ingroup_lang/'
#ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
SR_FOLDER_MONTH = ROOT + 'subreddits_month/'
SR_FOLDER = ROOT + 'subreddits/'
SR_FOLDER2 = ROOT + 'subreddits2/'
SUBREDDITS = DATA + 'subreddit_list.txt'
REMOVED_SRS = DATA + 'non_english_sr.txt'
USERS = LOGS + 'users/'

conf = SparkConf()
sc = SparkContext(conf=conf)
reddits = set()

def subreddit_of_interest(line): 
    comment = json.loads(line)
    return 'subreddit' in comment and 'body' in comment and \
        comment['subreddit'].lower() in reddits

def get_comment_user(line): 
    '''
    comment ID to author
    '''
    comment = json.loads(line)
    return (comment['id'], comment['author'].lower())

def get_comment_parent(line): 
    '''
    comment ID to parent ID
    '''
    comment = json.loads(line)
    return (comment['id'], comment['parent_id'])

def get_user_counts(line): 
    comment = json.loads(line)
    if 'subreddit' in comment and 'body' in comment: 
        return ((comment["subreddit"].lower(), comment['author'].lower()), 1)
    else: 
        return ((None, None),1)

def get_top_user_sets(tup): 
    '''
    Input is (subreddit, list of (user, count))
    '''
    percent_param = 0.2
    c = Counter()
    for pair in tup[1]: 
        c[pair[0]] = pair[1]
    num_top_p = int(percent_param*len(c))
    vocab = set()
    for w in c.most_common(num_top_p): 
        vocab.add(w[0])
    return (tup[0], vocab)

def users_of_interest(line, user_counts): 
    # get access to broadcast variable
    comment = json.loads(line)
    author = comment["author"].lower()
    subreddit = comment['subreddit'].lower()
    return author in user_counts.value[subreddit] 

def get_sr_comment(line): 
    comment = json.loads(line)
    return (comment['subreddit'].lower(), [comment['id']])
    
def create_network_inputs(): 
    log_file = open(LOGS + 'network.temp', 'w') 
    non_english_reddits = set()
    with open(REMOVED_SRS, 'r') as inputfile: 
        for line in inputfile: 
            non_english_reddits.add(line.strip().lower())
    with open(SUBREDDITS, 'r') as inputfile: 
        for line in inputfile: 
            sr = line.strip().lower()
            if sr not in non_english_reddits: 
                reddits.add(sr)
    global MONTH
    MONTH = 'RC_2019-05'
    path = DATA + MONTH
    #path = DATA + 'tinyData'
    data = sc.textFile(path)
    log_file.write("BEFORE USER FILTER:" + str(data.count()) + '\n')

    # filter by top subreddits and top users
    data = data.filter(subreddit_of_interest)
    user_rdd = data.map(get_user_counts)
    # (subreddit, user)->count
    user_rdd = user_rdd.reduceByKey(lambda n1, n2: n1 + n2)
    # subreddit -> (user, count)
    user_rdd = user_rdd.map(lambda tup: (tup[0][0], [(tup[0][1], tup[1])]))
    user_rdd = user_rdd.reduceByKey(lambda n1, n2: n1 + n2)
    log_file.write("USER RDD EXAMPLE" + str(user_rdd.take(1)) + '\n')
    user_rdd = user_rdd.map(get_top_user_sets)
    log_file.write("USER RDD EXAMPLE" + str(user_rdd.take(1)) + '\n')
    
    user_counts = sc.broadcast(user_rdd.collectAsMap())
    data = data.filter(partial(users_of_interest, user_counts=user_counts))
    log_file.write("AFTER USER FILTER:" + str(data.count()) + '\n')
    
    # save a dictionary of comment ID to author
    comment_author = data.map(get_comment_user)
    comment_author = comment_author.collectAsMap()
    with open(LOGS + 'commentID_author.json', 'w') as outfile: 
        json.dump(comment_author, outfile)
    log_file.write('done with comment_author\n')
    # save a dictionary of comment to parent ID 
    comment_parent = data.map(get_comment_parent)
    comment_parent = comment_parent.collectAsMap()
    with open(LOGS + 'commentID_parentID.json', 'w') as outfile: 
        json.dump(comment_parent, outfile)
    log_file.write('done with comment_parent\n')
    # save comment IDs for each subreddit
    sr_comments = data.map(get_sr_comment)
    sr_comments = sr_comments.reduceByKey(lambda n1, n2: n1 + n2)
    sr_comments = sr_comments.collectAsMap()
    with open(LOGS + 'sr_commentIDs.json', 'w') as outfile: 
        json.dump(sr_comments, outfile)
    log_file.write('done with subreddit_comment\n')
    log_file.close()
    
def main(): 
    create_network_inputs()

if __name__ == '__main__':
    main()
