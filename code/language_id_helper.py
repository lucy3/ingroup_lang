'''
We filter out non-English subreddits. 
'''

import json 
import shutil
LOG_DIR = '/data0/lucy/ingroup_lang/logs/'
DATA = '/data0/lucy/ingroup_lang/data/'
NOTENG = '/data0/lucy/ingroup_lang/nonenglish_sr/'
SR_MONTHS = '/data0/lucy/ingroup_lang/subreddits_month/'

def get_nonenglish_sr(): 
    with open(LOG_DIR + 'subreddit_langs.json', 'r') as infile: 
        d = json.load(infile)
    with open(DATA + 'non_english_sr.txt', 'w') as outfile: 
        for sr in d: 
            total = 0
            most_common_lang = ''
            most_common_count = 0
            for tup in d[sr]: 
                if tup[1] > most_common_count: 
                    most_common_lang = tup[0]
                    most_common_count = tup[1]
                total += tup[1]
            percent = most_common_count / float(total)
            if most_common_lang != u'en' or percent < 0.85: 
                print(sr, most_common_lang, percent)
                outfile.write(sr + '\n') 

def move_nonenglish_sr(): 
    non_english_subreddits = set()
    with open(DATA + 'non_english_sr.txt', 'r') as infile: 
        for line in infile: 
            non_english_subreddits.add(line.strip().lower())
    for sr in non_english_subreddits: 
        shutil.move(SR_MONTHS + sr, NOTENG + sr)

def main(): 
    get_nonenglish_sr()
    move_nonenglish_sr()
        
if __name__ == '__main__':
    main()
