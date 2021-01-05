"""
TextRank 

Mihalcea & Tarau 2004
"""

import networkx as nx
import sys
import os
import csv
from transformers import BasicTokenizer
import spacy
from spacy.tokens import Doc
import time

ROOT = '/mnt/data0/lucy/ingroup_lang/'
INFOLDER = ROOT + 'subreddits_month/'
OUTFOLDER = ROOT + 'logs/keywords_textrank/' 

class CustomTokenizer(object): 
    def __init__(self, vocab): 
        self.tokenizer = BasicTokenizer(do_lower_case=False)
        self.vocab = vocab
    def __call__(self, text): 
        words = self.tokenizer.tokenize(text.strip())
        return Doc(self.vocab, words=words)    

def main(): 
    """
    Undirected, unweighted graph with parameters from original Textrank paper
    """
    tokenizer = BasicTokenizer(do_lower_case=True)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    nlp.tokenizer = CustomTokenizer(nlp.vocab)
    for article in os.listdir(INFOLDER):
        start = time.time()
        print("creating graph for", article) 
        G = nx.Graph()

        infile = open(INFOLDER + article + '/RC_sample', 'r') 
        for line in infile: 
            if line.startswith('USER1USER0USER'): continue
            doc = nlp(line)
            comment_toks = []
            for token in doc: 
                if token.pos_ in set(['ADJ', 'NOUN', 'PROPN']): 
                     comment_toks.append(token.text.lower())
            
            # add edges to graph using window size of 2
            for i in range(len(comment_toks) - 1): 
                w1 = comment_toks[i]
                w2 = comment_toks[i + 1]
                if not G.has_edge(w1, w2):
                    G.add_edge(w1, w2)
        infile.close()
        print("TIME:", time.time() - start)
        print("calculating pagerank...")
        pagerank_scores = nx.pagerank(G, alpha=0.85, tol=0.0001)
        outfile = open(OUTFOLDER + article, 'w')
        writer = csv.writer(outfile)
        writer.writerow(['word', 'textrank'])
        vals = sorted(pagerank_scores.items(), key=lambda item: item[1])
        for tup in vals: 
            writer.writerow([tup[0], tup[1]])
        outfile.close()
        
        print("TOTAL TIME:", time.time() - start)
        break
            

if __name__ == "__main__":
    main()
