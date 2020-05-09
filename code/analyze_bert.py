"""
Note: just run Spark locally
"""

#from pyspark import SparkConf, SparkContext
import numpy as np
#from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from MulticoreTSNE import MulticoreTSNE as TSNE
import random
from collections import defaultdict

ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
VEC_FOLDER = ROOT + 'logs/senses_viz/'
LOGS = ROOT + 'logs/'

#conf = SparkConf()
#sc = SparkContext(conf=conf)

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

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

def compare_word_across_subreddits(subreddit_list, word, finetuned=False): 
    '''
    This implementation assumes that we saved individual files
    for each word and each subreddit (since we don't want to
    waste time saving every vector, only ones we care about
    for some visualization)
    '''
    #marker_options = ['o', 's', 's', 'D', 'D', 'D', 'P', '*', '^', '>']
    marker_options = ['o', 's', 'D', 'P', '*', '^', '>']
    #color_options = ['tab:blue', 'tab:orange', 'tab:green', 
    #     'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 
    #     'tab:gray', 'tab:olive', 'tab:cyan']
    color_options = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    srs = defaultdict(list) # subreddit name : indices
    sense_colors = []
    X = []
    j = 0
    for sr in subreddit_list: 
        sense_path = VEC_FOLDER + sr + '_' + str(finetuned)
        with open(sense_path, 'r') as infile: 
            for line in infile:
                # speeds up visualization if we only visualize a sample  
                if random.choice(range(4)) != 0: continue
                contents = line.strip().split('\t') 
                w = contents[1]
                if word != w: continue
                sense_colors.append(int(contents[2]))
                rep = np.array([float(i) for i in contents[3].split()])
                X.append(rep)
                srs[sr].append(j)
                j += 1
    sense_colors = np.array(sense_colors) 
    X = np.array(X)
    X_embedded = TSNE(n_components=2, n_jobs=-1, random_state=0).fit_transform(X)
    fig, ax = plt.subplots()
    legend_elements = [] 
    for i, sr in enumerate(subreddit_list): 
        idx = srs[sr]
        # can put c=color_options[i] if we want to color by subreddit        
        ax.scatter(X_embedded[idx,0], X_embedded[idx,1], c=color_options[i], \
               marker=marker_options[i], alpha=0.5, s=10, linewidths=0)
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=sr, 
                                   markerfacecolor=color_options[i], markersize=10))
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5,-0.1), ncol=3)
    plt.savefig('../logs/bert_reddit_' + word + '_subreddits_' + str(finetuned) + '.png', 
        bbox_inches="tight", dpi=300) 
    plt.close()
    fig, ax = plt.subplots()
    for i, sr in enumerate(subreddit_list): 
        idx = srs[sr]
        colors = [color_options[k] for k in sense_colors[idx]]
        ax.scatter(X_embedded[idx,0], X_embedded[idx,1], c=colors, \
               marker='.', alpha=0.5) 
        # can put c=colors if we want to color by sense 
    plt.savefig('../logs/bert_reddit_' + word + '_senses_' + str(finetuned) + '.png', 
        bbox_inches="tight")
    plt.close()
     

def compare_word_across_subreddits_old(subreddit_list, word): 
    '''
    This implementation assumes that each subreddit has all of
    its vectors saved and we must filter them.
    '''
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
        data = sc.textFile(ALL_VEC_FOLDER + sr)
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

def get_lemma_vectors(x, lemma, dataset): 
    contents = x.strip().split('\t') 
    token = contents[0].split('_')[-1]
    l = contents[0].split('_')[-2]
    if dataset == 'semeval2010': 
        pos = l.split('.')[1]
        return token == wnl.lemmatize(contents[1], pos) and l == lemma
    elif dataset == 'semeval2013': 
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

def plot_semeval_clusters(lemma, dataset, cluster_path): 
    """
    Plot gold and my clusters on the same plot
    Shapes are gold clusters
    Colors are my clusters
    """
    marker_options = ['o', 'v', '^', '<', '>', 's', 'P', '*', 'X', 'D']
    color_options = ['tab:blue', 'tab:orange', 'tab:green', 
         'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 
         'tab:gray', 'tab:olive', 'tab:cyan']
    if dataset == 'semeval2013': 
        infile = LOGS + 'semeval2013/semeval2013_test_bert3'
        goldpath = ROOT + 'SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key'
    elif dataset == 'semeval2010': 
        infile = LOGS + 'semeval2010/semeval2010_test_bert2'
        goldpath = ROOT + 'semeval-2010-task-14/evaluation/unsup_eval/keys/all.key'
    data = sc.textFile(infile) 
    data = data.filter(lambda x: get_lemma_vectors(x, lemma, dataset))
    data = data.map(get_instance_vectors)
    vecs = data.collectAsMap()
    colors = []
    markers = []
    labels1 = []
    labels2 = []
    gold_clusters = defaultdict(list)
    my_labels = {}
    with open(goldpath, 'r') as infile: 
        for line in infile:  
            contents = line.strip().split()
            instance = contents[1]
            if contents[0] != lemma: continue
            label = contents[2]
            if label not in labels1: 
                labels1.append(label)
            i = labels1.index(label)
            gold_clusters[i].append(instance)
    with open(LOGS + dataset + '/' + cluster_path, 'r') as infile: 
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
    plt.savefig('../logs/bert_' + lemma + '_' + dataset + '_' + cluster_path + '.png')

def main():
    #sanity_check()
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], '!')
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], '.')
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], 'the')
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], 'london')

    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], 'fire')
    #compare_word_across_subreddits(['vegan', 'financialindependence', 'fashionreps', 'keto', 'applyingtocollege'], 'sick')
    #compare_semeval2013_lemmas('add.v', test=True)
    #plot_semeval_clusters('house.n', 'semeval2010', 'semeval2010_clusters20_0')
    #plot_semeval_clusters('house.n', 'semeval2010', 'semeval2010_clusters3_0')
    #plot_semeval_clusters('house.n', 'semeval2010', 'semeval2010_clusters3_0_normalize') 
    compare_word_across_subreddits(['warriors', 
         'nba', 'cooking'], 'curry', finetuned=True)
    compare_word_across_subreddits(['hardwareswap', 'mechmarket', 'ukpolitics', 
         'skincareaddiction'], 'pm', finetuned=True)
    compare_word_across_subreddits(['walmart', 'magicarena', 
         'datingoverthirty'], 'spark', finetuned=True)
    #compare_word_across_subreddits(['boxoffice', 'overwatch', 
    #     'competitiveoverwatch', 'repsneakers', 'sneakers', 'fashionreps'], 'ow', finetuned=True)
    #compare_word_across_subreddits(['borderlands', 'swgalaxyofheroes', 'reddeadredemption', 
    #   'reddeadonline', 'starwars', 'starwars'], 'hunters', finetuned=True)
    #compare_word_across_subreddits(['borderlands', 'sekiro', 'boardgames', 
    #   'thedivision', 'destinythegame'], 'hunters', finetuned=False)
    #sc.stop()

if __name__ == '__main__': 
    main()
