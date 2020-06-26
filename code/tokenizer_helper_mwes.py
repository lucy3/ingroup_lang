'''
Tokenizes using BERT BasicTokenizer
and removes duplicates for each user.

Use this instead of stanford_tokenizer.sh and remove_duplicates.sh
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
    with open(ROOT + 'logs/all_mwes.json', 'r') as infile:
        mwe_dict = json.load(infile)
    with open(filename, 'r') as infile: 
        for line in infile: 
            if line.startswith('USER1USER0USER'): 
                if len(vocab) > 0: 
                    outfile.write('\n'.join(list(vocab)) + '\n')  
                vocab = set()
            else: 
                tokens = tokenizer.tokenize(line.strip())
                new_tokens = []
                i = 0
                while i < len(tokens): 
                    w = tokens[i]
                    if w in mwe_dict and i + 1 < len(tokens): 
                       curr_layer = mwe_dict[w]
                       curr = i + 1
                       res = w + ' '
                       poss_complete = None
                       poss_i = curr
                       while curr < len(tokens) and tokens[curr] in curr_layer:
                          if '$END_TOKENS$' in curr_layer: 
                              poss_complete = res
                              poss_i = curr
                          res += tokens[curr] + ' ' 
                          curr_layer = curr_layer[tokens[curr]] 
                          curr += 1
                       if '$END_TOKEN$' in curr_layer: 
                          poss_complete = res
                          poss_i = curr
                       if poss_complete is not None: 
                          i = poss_i
                          new_tokens.append(poss_complete.strip())
                       else: 
                          i += 1
                          new_tokens.append(w)
                    else: 
                       i += 1
                       new_tokens.append(w)
                # for each word, check if it's the beginning of a mwe
                # the MWE dictionary is of format word1 -> word2 -> $END_TOKEN$
                # if the next word isn't part of the dictionary, check for end token
                # the end token might appear in an earlier branch 
                vocab.update(new_tokens)
        if len(vocab) > 0: 
            outfile.write('\n'.join(list(vocab)) + '\n')
    outfile.close()    

if __name__ == "__main__":
    main()
