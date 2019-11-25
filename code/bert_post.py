"""
Piece together wordpiece vectors
"""
import numpy as np
import sys

def merge_wordpiece(inpath, outpath): 
    outfile = open(outpath, 'w')
    with open(inpath, 'r') as infile: 
        ongoing_word = []
        ongoing_word_rep = []
        prev_line = None
        for line in infile: 
            contents = line.strip().split('\t') 
            ID = contents[0]
            token = contents[1]
            if not token.startswith('##'): 
                if len(ongoing_word) == 0 and prev_line is not None: 
                    # write previous line
                    outfile.write(prev_line)
                elif prev_line is not None:
                    # join stuff together
                    rep = np.array(ongoing_word_rep)
                    rep = np.mean(rep, axis=0).flatten()
                    contents_prev = prev_line.strip().split('\t') 
                    prev_ID = contents_prev[0]
                    prev_token = contents_prev[1]
                    tok = ''
                    for t in ongoing_word: 
                        if t.startswith('##'): t = t[2:]
                        tok += t
                    outfile.write(prev_ID + '\t' + tok + '\t' + ' '.join([str(j) for j in rep]) + '\n') 
                ongoing_word = []
                ongoing_word_rep = []
            else:
                if len(ongoing_word) == 0 and prev_line is not None: 
                    # add previous line to ongoing stuff
                    contents_prev = prev_line.strip().split('\t') 
                    prev_ID = contents_prev[0]
                    prev_token = contents_prev[1]
                    prev_rep = [float(j) for j in contents_prev[2].split()]
                    ongoing_word.append(prev_token)
                    ongoing_word_rep.append(prev_rep)
                ongoing_word.append(token)
                ongoing_word_rep.append([float(j) for j in contents[2].split()]) 
            prev_line = line
        # fence post
        if len(ongoing_word) == 0 and prev_line is not None: 
            # write previous line
            outfile.write(prev_line)
        elif prev_line is not None:
            # join stuff together
            rep = np.array(ongoing_word_rep)
            rep = np.mean(rep, axis=0).flatten()
            contents_prev = prev_line.strip().split('\t') 
            prev_ID = contents_prev[0]
            prev_token = contents_prev[1]
            tok = ''
            for t in ongoing_word: 
                if t.startswith('##'): t = t[2:]
                tok += t
            outfile.write(prev_ID + '\t' + tok + '\t' + ' '.join([str(j) for j in rep]) + '\n') 
    outfile.close()
            

def main(): 
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    merge_wordpiece(inpath, outpath)

if __name__ == "__main__":
    main()
