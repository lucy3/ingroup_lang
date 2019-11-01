"""
Remove duplicate words for
each user after
lowercasing every word 
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
            if w not in vocab: 
                vocab.add(w)
                outfile.write(w + '\n') 
    outfile.close()

if __name__ == '__main__':
    main()
