import os
from collections import Counter
import matplotlib
import numpy as np 
matplotlib.use('agg')
import matplotlib.pyplot as plt

ROOT = '/data0/lucy/ingroup_lang/'
SR_FOLDER_MONTH = ROOT + 'subreddits_month/'
LOGS = ROOT + 'logs/'

def count_tokens(): 
    pass          
    
def count_comments(): 
    comment_count = Counter()
    for sr in os.listdir(SR_FOLDER_MONTH): 
        num_lines = 0
        with open(SR_FOLDER_MONTH + sr + '/RC_sample', 'r') as infile: 
            for line in infile: 
                if line.startswith('@@#USER#@@_'): continue
                num_lines += 1
        comment_count[sr] = num_lines
    with open(LOGS + 'dataset_statistics_comments.txt', 'w') as outfile: 
        for tup in comment_count.most_common(): 
            outfile.write(tup[0] + '\t' + str(tup[1]) + '\n')
    '''
    # the below code doesn't work
    plt.xscale('log')
    plt.hist(comment_count.values(), bins=np.logspace(np.log10(0.1),np.log10(1.0), 10))
    plt.ylabel("number of subreddits") 
    plt.xlabel("total # of comments") 
    plt.savefig(LOGS + 'dataset_statistics_comments.png')
    '''
        

def main(): 
    count_tokens()
    count_comments()

if __name__ == "__main__":
    main()
