"""
Remove duplicate words for each user in
tokenized file. This means if we see a line that starts
with USER1USER0USER, we create a new set. We ignore username
lines as well when outputting to a new file, and lowercase all words. 

This file should be parallelized across subreddits if possible
and written similarly to rm_dups_single_user.py 
"""
import sys

def main(): 
    filename = sys.argv[1]
    outpath = filename.replace('subreddits2', 'subreddits3')
    outfile = open(outpath, 'w') 
    vocab = set()
    with open(filename, 'r') as infile: 
        for line in infile: 
            w = line.strip().lower()
            if line.startswith('USER1USER0USER'): 
                vocab = set()
            elif w not in vocab: 
                vocab.add(w)
                outfile.write(w + '\n') 
    outfile.close()    

if __name__ == "__main__":
    main()
