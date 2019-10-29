"""
Note: just run Spark locally
"""

from pyspark import SparkConf, SparkContext
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VEC_FOLDER = '/global/scratch/lucy3_li/ingroup_lang/logs/bert_vectors/'

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
    s = ax.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.5)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='soy',
                          markerfacecolor='g', markersize=10), 
                       Line2D([0], [0], marker='o', color='w', label='dish',
                          markerfacecolor='r', markersize=10),]
    ax.legend(handles=legend_elements)
    plt.savefig('../logs/bert_sanity_check_' + subreddit + '.png')

def compare_word_across_subreddits(subreddit_list): 
    X = []
    word = 'fire'
    color_map = {}
    color_map_rev = {}
    for sr in subreddit_list: 
        color = np.random.rand(3,)
        color_map[sr] = color
        color_map_rev[color] = sr 
        data = sc.textfile(VEC_FOLDER + sr)
        data = data.filter(lambda x: get_word_subset(x, word))
        data = data.map(get_word_vectors)
        vecs = data.collect()
        for item in vecs: 
            words.append(item[0])
            X.append(item[1])
            colors.append(color_map[sr])
    print("WORDS:", set(words))
    X = np.array(X)
    X_embedded = TSNE(n_components=2).fit_transform(X)
    print ("POST-TSNE SHAPE:", X_embedded.shape)
    fig, ax = plt.subplots()
    ax.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, alpha=0.5)
    legend_elements = []
    for color in color_map: 
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=color_map_rev[color], 
                                   markerfacecolor=color, markersize=10))
    ax.legend(handles=legend_elements)
    plt.savefig('../logs/bert_viz_single_word_' + word + '.png')
 

def main():
    #sanity_check()
    compare_words_across_subreddits(['vegan', 'financialindependence'])
    sc.stop()

if __name__ == '__main__': 
    main()
