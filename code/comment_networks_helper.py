"""
Functions for direct-reply networks
"""
from collections import Counter
import json
import networkx as nx

ROOT = '/data0/lucy/ingroup_lang/'
#ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
NETWORKS = LOGS + 'networks/'

def calculate_density(): 
    '''
    calculate density of
    the network
    '''
    outfile = open(LOG_DIR + 'commentor_density', 'w')
    outfile.write('subreddit,density\n')
    for sr in os.listdir(NETWORKS): 
        G = nx.read_edgelist(NETWORKS + sr, delimiter=',', data=(('weight',float),))
        print(type(G))
        break
        outfile.write(sr + ',' + str(nx.density(G)) + '\n')
    outfile.close()

def create_edgelist(): 
    '''
    draw edge between author and parent's author if parent
    exists in commentID->author dictionary
    use a counter with sorted key (author, author)
    save a csv of author,author and weight for each subreddit
    
    The parent ID has a prefix "t1_" in front of it
    or 't3_' if the parent is a post. 
    '''
    with open(LOGS + 'commentID_author.json', 'r') as infile: 
        comment_author = json.load(infile)
    with open(LOGS + 'commentID_parentID.json', 'r') as infile: 
        comment_parent = json.load(infile)
    with open(LOGS + 'sr_commentIDs.json', 'r') as infile: 
        sr_comments = json.load(infile)
    edges = Counter()
    for sr in sr_comments: 
        comments = sr_comments[sr]
        for commentID in comments: 
            author1 = comment_author[commentID]
            parentID = comment_parent[commentID]
            if parentID[3:] in comment_author: 
                author2 = comment_author[parentID[3:]]
                edges[tuple(sorted([author1, author2]))] += 1
        with open(NETWORKS + sr, 'w') as outfile: 
            for e in edges: 
                outfile.write(e[0] + ',' + e[1] + ',' + str(edges[e]) + '\n') 

def main(): 
    create_edgelist()

if __name__ == '__main__':
    main()
