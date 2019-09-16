"""
Python 3
"""
import os
from stanfordnlp.server import CoreNLPClient
import time

ROOT = '/data0/lucy/ingroup_lang/'
LOG_DIR = ROOT + 'logs/'
SR_FOLDER = ROOT + 'subreddits/'
SR_FOLDER2 = ROOT + 'subreddits2/'

def main(): 
    '''
    Time it takes to run
    - line by line: 400.775
    - 
    '''
    start = time.time()
    MONTH = 'RC_2019-05'
    log_file = open(LOG_DIR + 'tokenize.temp', 'w')
    # max_char_length = -1 doesn't work?
    with CoreNLPClient(annotators=['tokenize','ssplit'], timeout=50000000, 
                       threads=20, memory='64G') as client:
        for folder_name in os.listdir(SR_FOLDER): 
            if os.path.isdir(SR_FOLDER + folder_name):
                print(folder_name)
                log_file.write(folder_name + '\n') 
                path = SR_FOLDER + folder_name + '/' + MONTH
                outfile = open(SR_FOLDER2 + folder_name + '_' + MONTH + '.txt', 'w')
                with open(path, 'r') as infile: 
                    for line in infile: 
                        ann = client.annotate(line)
                        for s in ann.sentence: 
                            for t in s.token: 
                                outfile.write(t.word.lower() + ' ') 
                        outfile.write('\n')
                outfile.close()
                log_file.write('\t DONE' + '\n') 
    log_file.close()
    print(time.time() - start) 
            
            
if __name__ == '__main__':
    main()
