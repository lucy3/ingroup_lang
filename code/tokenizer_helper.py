'''
Tokenizes using BERT BasicTokenizer
and removes duplicates for each user.

Called by tokenizer.sh 
'''

from transformers import BasicTokenizer
import sys
import json 

ROOT = '/mnt/data0/lucy/ingroup_lang/' 

def main(): 
    filename = sys.argv[1]
    outpath = filename.replace('subreddits_month', 'subreddits3') + '.conll'
    outfile = open(outpath, 'w') 
    tokenizer = BasicTokenizer(do_lower_case=True)
    vocab = set()
    with open(filename, 'r') as infile: 
        for line in infile: 
            if line.startswith('USER1USER0USER'): 
                if len(vocab) > 0: 
                    outfile.write('\n'.join(list(vocab)) + '\n')  
                vocab = set()
            else: 
                tokens = tokenizer.tokenize(line.strip())
                vocab.update(tokens)
        if len(vocab) > 0: 
            outfile.write('\n'.join(list(vocab)) + '\n')
    outfile.close()    

if __name__ == "__main__":
    main()
