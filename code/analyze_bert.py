"""
Note: just run Spark locally
"""

from pyspark import SparkConf, SparkContext
import numpy as np
#from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from MulticoreTSNE import MulticoreTSNE as TSNE
import random
from collections import defaultdict

ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
VEC_FOLDER = ROOT + 'logs/bert_vectors/'
LOGS = ROOT + 'logs/'

conf = SparkConf()
sc = SparkContext(conf=conf)

def get_word_subset(line, w): 
    contents = line.strip().split('\t') 
    word = contents[1]
    return word == w

def get_word_vectors(line): 
    contents = line.strip().split('\t') 
    word = contents[1]
    vec = np.array([float(i) for i in contents[2].split()])
    return (word, vec)

def get_instance_vectors(line): 
    contents = line.strip().split('\t') 
    ID = contents[0]
    small_id = ID.split('_')[-3]
    vec = np.array([float(i) for i in contents[2].split()])
    return (small_id, vec)

def sanity_check():
    print("RUNNING SANITY CHECK") 
    subreddit = 'vegan'
    path = VEC_FOLDER + subreddit
    data = sc.textFile(path)
    print("NROWS OF DATA", data.count())
    data1 = data.filter(lambda x: get_word_subset(x, 'dish'))
    data1 = data1.map(get_word_vectors)
    vecs1 = data1.collect()
    data2 = data.filter(lambda x: get_word_subset(x, 'soy'))
    data2 = data2.map(get_word_vectors)
    vecs2 = data2.collect()
    print("NUMBER OF VECTORS:", len(vecs1) + len(vecs2))
    print("EXAMPLE OF ONE ITEM IN VECS:", vecs1[0])
    color_map = {'r': 'dish', 'g': 'soy'}
    words = []
    X = []
    colors = []
    for item in vecs1: 
        words.append(item[0])
        X.append(item[1])
        colors.append('r')
    for item in vecs2: 
        words.append(item[0])
        X.append(item[1])
        colors.append('g')
    print("WORDS:", set(words))
    X = np.array(X)
    X_embedded = TSNE(n_components=2).fit_transform(X)
    print('POST-TSNE SHAPE:', X_embedded.shape)
    fig, ax = plt.subplots()
    s = ax.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.5, marker='.')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='soy',
                          markerfacecolor='g', markersize=10), 
                       Line2D([0], [0], marker='o', color='w', label='dish',
                          markerfacecolor='r', markersize=10),]
    ax.legend(handles=legend_elements)
    plt.savefig('../logs/bert_sanity_check_' + subreddit + '.png')

def compare_word_across_subreddits(subreddit_list, word): 
    X = []
    colors = []
    color_map = {}
    color_map_rev = {}
    color_options = ['b', 'g', 'r', 'y', 'c', 'm']
    print("GETTING VECTORS FOR WORD:", word)
    for i, sr in enumerate(subreddit_list): 
        color = color_options[i]
        color_map[sr] = color
        color_map_rev[color] = sr 
        data = sc.textFile(VEC_FOLDER + sr)
        data = data.filter(lambda x: get_word_subset(x, word))
        total = data.count()
        if total > 500: 
            data = data.sample(False, 500.0/total, 0)
        data = data.map(get_word_vectors)
        vecs = data.collect()
        for item in vecs: 
            X.append(item[1])
            colors.append(color_map[sr])
    X = np.array(X)
    X_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(X)
    print("POST-TSNE SHAPE:", X_embedded.shape)
    fig, ax = plt.subplots()
    ax.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.3, marker='.')
    legend_elements = []
    for color in color_map_rev: 
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=color_map_rev[color], 
                                   markerfacecolor=color, markersize=10))
    ax.legend(handles=legend_elements)
    if word == '.': word = 'period'
    if word == '!': word = 'exclaimmark'
    plt.savefig('../logs/bert_viz_single_word_' + word + '.png')

def get_lemma_vectors(x, lemma): 
    contents = x.strip().split('\t') 
    token = contents[0].split('_')[-1]
    l = contents[0].split('_')[-2]
    return token == contents[1] and l == lemma

def compare_semeval2013_lemmas(lemma, test=False): 
    X = []
    colors = []
    words = []
    if test: 
        infile = LOGS + 'semeval2013_test_bert3'
    else: 
        infile = LOGS + 'semeval2013_bert'
    data = sc.textFile(infile) 
    data = data.filter(lambda x: get_lemma_vectors(x, lemma))
    data = data.map(get_word_vectors)
    vecs = data.collect()
    for item in vecs: 
        X.append(item[1])
        words.append(item[0]) 
    vocab = set(words)
    color_map = {}
    color_map_rev = {}
    for w in vocab: 
        color_map_rev[tuple(np.random.rand(3,))] = w
        color_map[w] = tuple(np.random.rand(3,))
    colors = []
    for w in words: 
        colors.append(color_map[w])
    X = np.array(X)
    X_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(X)
    print("POST-TSNE SHAPE:", X_embedded.shape)
    fig, ax = plt.subplots()
    ax.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.3, marker='.')
    legend_elements = []
    for color in color_map_rev: 
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=color_map_rev[color], 
                                   markerfacecolor=color, markersize=10))
    ax.legend(handles=legend_elements)
    if test: 
        plt.savefig('../logs/bert_viz_single_lemma_' + lemma + '_semeval_test.png')
    else: 
        plt.savefig('../logs/bert_viz_single_lemma_' + lemma + '_semeval.png')

def plot_semeval_clusters(lemma): 
    """
    Plot gold and my clusters on the same plot
    """
    marker_options = ['o', 'v', '^', '<', '>', 's', 'P', '*', 'X', 'D']
    color_options = ['tab:blue', 'tab:orange', 'tab:green', 
         'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 
         'tab:gray', 'tab:olive', 'tab:cyan']
    infile = LOGS + 'semeval2013_test_bert3'
    data = sc.textFile(infile) 
    data = data.filter(lambda x: get_lemma_vectors(x, lemma))
    data = data.map(get_instance_vectors)
    vecs = data.collectAsMap()
    colors = []
    markers = []
    labels1 = []
    labels2 = []
    gold_clusters = defaultdict(list)
    my_labels = {}
    goldpath = ROOT + 'SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key'
    with open(goldpath, 'r') as infile: 
        for line in infile:  
            contents = line.strip().split()
            instance = contents[1]
            if contents[0] != lemma: continue
            label = contents[2].split('/')[0]
            if label not in labels1: 
                labels1.append(label)
            i = labels1.index(label)
            gold_clusters[i].append(instance)
    with open(LOGS + 'semeval_test_clusters', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split()
            instance = contents[1]
            if contents[0] != lemma: continue
            label = contents[2] + contents[0]
            if label not in labels2: 
                labels2.append(label)
            i = labels2.index(label)
            my_labels[instance] = i
    print("NUM CLUSTERS VS NUM COLORS", len(labels2), len(color_options))
    X = []
    instances = []
    for k in vecs: 
        instances.append(k)
        X.append(vecs[k])
    X = np.array(X)
    X_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(X)
    fig, ax = plt.subplots()
    for i in gold_clusters: 
        inst = gold_clusters[i]
        idx = []
        colors = []
        for k in inst: 
            idx.append(instances.index(k))
            colors.append(color_options[my_labels[k]])
        ax.scatter(X_embedded[idx,0], X_embedded[idx,1], c=colors, marker=marker_options[i])
    plt.savefig('../logs/bert_' + lemma + '_semeval2013_test.png')

def main():
    #sanity_check()
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], '!')
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], '.')
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], 'the')
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], 'london')

    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], 'fire')
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], 'sick')
    #compare_semeval2013_lemmas('add.v', test=True)
    plot_semeval_clusters('add.v') 
    sc.stop()

if __name__ == '__main__': 
    main()
