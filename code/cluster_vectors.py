from pyspark import SparkConf, SparkContext
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import bcubed
from collections import defaultdict

conf = SparkConf()
sc = SparkContext(conf=conf)

ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
LOGS = ROOT + 'logs/'
SEMEVAL_VECTORS = LOGS + 'semeval2013_bert'
SEMEVAL_TEST_VECTORS = LOGS + 'semeval2013_test_bert'

def semeval_words_of_interest(line): 
    contents = line.strip().split('\t')
    token = contents[0].split('_')[-1]
    return token == contents[1]

def get_semeval_vector(line): 
    contents = line.strip().split('\t') 
    ID = contents[0]
    token = ID.split('_')[-2] # lemma
    vector = np.array([[float(i) for i in contents[2].split()]])
    return (token, ([ID], vector))

def kmeans_with_gap_statistic(tup): 
    """
    Based off of https://anaconda.org/milesgranger/gap-statistic/notebook
    """
    token = tup[0]
    IDs = tup[1][0]
    data = tup[1][1]
    nrefs = 50
    ks = range(2, 10)
    gaps = np.zeros(len(ks))
    labels = {} # k : km.labels_
    s = np.zeros(len(ks))
    for i, k in enumerate(ks):
        ref_disps = np.zeros(nrefs)
        for j in range(nrefs):
            random_ref = np.random.random_sample(size=data.shape)
            km = KMeans(k, n_jobs=-1)
            km.fit(random_ref)
            ref_disps[j] = km.inertia_
        km = KMeans(k, n_jobs=-1)
        km.fit(data)
        orig_disp = km.inertia_
        gap = np.mean(np.log(ref_disps)) - np.log(orig_disp)
        s[i] = math.sqrt(1.0 + 1.0/nrefs)*np.std(ref_disps)
        gaps[i] = gap
        labels[k] = km.labels_
    for i in range(len(ks) - 1): 
        k = ks[i] 
        if gaps[i] >= gaps[i+1] + s[i+1]: 
            return labels[k]
    return (IDs, labels[ks[-1]])

def get_data_size(tup): 
    token = tup[0]
    data = tup[1][1]
    return (token, data.shape)
    
def semeval_clusters(test=False): 
    if test: 
        data = sc.textFile(SEMEVAL_TEST_VECTORS) 
    else: 
        data = sc.textFile(SEMEVAL_VECTORS)
    data = data.filter(semeval_words_of_interest)
    data = data.map(get_semeval_vector)
    data = data.reduceByKey(lambda n1, n2: (n1[0] + n2[0], np.concatenate((n1[1], n2[1]), axis=0)))
    #size = data.map(get_data_size).collectAsMap()
    data = data.map(kmeans_with_gap_statistic)
    clustered_IDs = data.collect()
    sc.stop()
    #for token in size: 
    #    print(token, size[token])
    if test: 
        outname = 'semeval_test_clusters'
    else: 
        outname = 'semeval_clusters'
    with open(LOGS + outname, 'w') as outfile: 
        for tup in clustered_IDs: 
            IDs = tup[0]
            labels = tup[1]
            for i, ID in enumerate(IDs): 
                small_id = ID.split('_')[-3]
                lemma = ID.split('_')[-2]
                outfile.write(lemma + ' ' + small_id + ' ' + str(labels[i]) + '\n')

def evaluate_nmi(): 
    sc.stop()
    print("calculating nmi...") 
    goldpath = ROOT + 'SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key'
    labels1 = []
    labels2 = []
    gold_labels = {}
    my_labels = {}
    with open(goldpath, 'r') as infile: 
        for line in infile:  
            contents = line.strip().split()
            instance = contents[1]
            label = contents[2].split('/')[0]
            if label not in labels1: 
                labels1.append(label)
            i = labels1.index(label)
            gold_labels[instance] = i
    with open(LOGS + 'semeval_test_clusters', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split()
            instance = contents[1]
            label = contents[2] + contents[0]
            if label not in labels2: 
                labels2.append(label)
            i = labels2.index(label)
            my_labels[instance] = i
    gold = []
    mine = []
    for k in gold_labels:
        if k not in my_labels: 
            print(k)
            continue
        gold.append(gold_labels[k])
        mine.append(my_labels[k])
    print("NMI:", normalized_mutual_info_score(gold, mine)) 

def evaluate_bcubed(): 
    print("calculating bcubed...")
    gold_labels = {}
    my_labels = {}
    labels = []
    goldpath = ROOT + 'SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key'
    gold_c = defaultdict(list)
    my_c = defaultdict(list)
    with open(goldpath, 'r') as infile: 
        for line in infile:  
            contents = line.strip().split()
            instance = contents[1]
            label = contents[2].split('/')[0]
            if label not in labels: 
                labels.append(label)
            i = labels.index(label)
            gold_labels[instance] = set([i])
            gold_c[i].append(instance) 
    with open(LOGS + 'semeval_test_clusters', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split()
            instance = contents[1]
            label = contents[2] + contents[0] # label is cluster # + lemma 
            if instance not in gold_labels: continue # only evaluate on things in gold
            my_labels[instance] = set([label])
            my_c[label].append(instance) 
    print("num gold clusters", len(gold_c), "num my clusters", len(my_c))
    precision = bcubed.precision(my_labels, gold_labels)
    recall = bcubed.recall(my_labels, gold_labels)
    fscore = bcubed.fscore(precision, recall)
    print("Precision:", precision, "Recall:", recall, "F-score:", fscore) 

def main(): 
    #semeval_clusters(test=True)
    evaluate_nmi()
    evaluate_bcubed()

if __name__ == "__main__":
    main()
