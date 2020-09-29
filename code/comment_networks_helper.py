"""
Functions for direct-reply networks
"""
from collections import Counter
import json
import networkx as nx
import os

ROOT = '/mnt/data0/lucy/ingroup_lang/'
#ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
NETWORKS = LOGS + 'networks/'

def get_user_centrality(): 
    '''
    Calculate betweenness centrality of each user in a subreddit
    '''
    centralities = {}
    for sr in os.listdir(NETWORKS):
        print(sr) 
        G = nx.read_edgelist(NETWORKS + sr, delimiter=',', data=(('weight',float),))
        G.remove_node("[deleted]")
        for e in G.edges(data=False): 
            G[e[0]][e[1]]['distance'] = 1.0 / G[e[0]][e[1]]['weight']
        
        #d = nx.eigenvector_centrality(G, weight='weight', max_iter=10000)
        #d = nx.closeness_centrality(G, distance='distance')
        centralities[sr] = d
    with open(LOGS + 'user_centralities.json', 'w') as outfile: 
        json.dump(centralities, outfile)
 
def calculate_density(): 
    '''
    calculate density of
    the network
    '''
    outfile = open(LOGS + 'commentor_density', 'w')
    outfile.write('subreddit,density\n')
    i = 0 
    for sr in os.listdir(NETWORKS): 
        G = nx.read_edgelist(NETWORKS + sr, delimiter=',', data=(('weight',float),))
        G.remove_node("[deleted]")
        outfile.write(sr + ',' + str(nx.density(G)) + '\n')
        i += 1
        if i % 100 == 0: print("Finished " + str(i) + " subreddits.") 
    outfile.close()
    
def sanity_check_edgelist(): 
    with open(LOGS + 'commentID_author.json', 'r') as infile: 
        comment_author = json.load(infile)
    with open(LOGS + 'commentID_parentID.json', 'r') as infile: 
        comment_parent = json.load(infile)
    with open(LOGS + 'sr_commentIDs.json', 'r') as infile: 
        sr_comments = json.load(infile)
    comments = sr_comments['whatsthisplant']
    for commentID in comments: 
        author1 = comment_author[commentID]
        if author1 == u'thebosspineapple' or author1 == u'thinkintime': 
            parentID = comment_parent[commentID]
            if parentID[3:] in comment_author: 
                author2 = comment_author[parentID[3:]]
                if author2 == u'thebosspineapple' or author2 == u'thinkintime': 
                    print(author1, commentID, author2, parentID) 

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
    for sr in sr_comments:
        edges = Counter()
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
    #create_edgelist()
    #calculate_density()
    #sanity_check_edgelist()
    get_user_centrality()

if __name__ == '__main__':
    main()
