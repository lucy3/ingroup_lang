from pyspark import SparkConf, SparkContext
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import bcubed
from collections import defaultdict, Counter
import json
import os
import xml.etree.ElementTree as ET
import re
import string
import spacy

ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
LOGS = ROOT + 'logs/'
SEMEVAL_VECTORS = LOGS + 'semeval2013_bert'
SEMEVAL_TEST_VECTORS = LOGS + 'semeval2013_test_bert2'
SEMEVAL_TEST_VECTORS2 = LOGS + 'semeval2013_test_bert3'

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
    lemma = tup[0]
    IDs = tup[1][0]
    data = tup[1][1]
    nrefs = 50
    ks = range(2, 10)
    gaps = np.zeros(len(ks))
    labels = {} # k : km.labels_
    centroids = {} 
    s = np.zeros(len(ks))
    for i, k in enumerate(ks):
        ref_disps = np.zeros(nrefs)
        for j in range(nrefs):
            random_ref = np.random.random_sample(size=data.shape)
            km = KMeans(k, n_jobs=-1, random_state=0)
            km.fit(random_ref)
            ref_disps[j] = km.inertia_
        km = KMeans(k, n_jobs=-1, random_state=0)
        km.fit(data)
        orig_disp = km.inertia_
        gap = np.mean(np.log(ref_disps)) - np.log(orig_disp)
        s[i] = math.sqrt(1.0 + 1.0/nrefs)*np.std(np.log(ref_disps))
        gaps[i] = gap
        labels[k] = km.labels_
        centroids[k] = km.cluster_centers_
    for i in range(len(ks) - 1): 
        k = ks[i] 
        if gaps[i] >= gaps[i+1] - s[i+1]:
            return (IDs, (labels[k], centroids[k]))
    return (IDs, (labels[ks[-1]], centroids[ks[-1]]))

def get_data_size(tup): 
    token = tup[0]
    data = tup[1][1]
    return (token, data.shape)
    
def semeval_clusters(test=False): 
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    if test: 
        data = sc.textFile(SEMEVAL_TEST_VECTORS2) 
    else: 
        data = sc.textFile(SEMEVAL_VECTORS)
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
            labels = tup[1][0]
            for i, ID in enumerate(IDs): 
                small_id = ID.split('_')[-3]
                lemma = ID.split('_')[-2]
                outfile.write(lemma + ' ' + small_id + ' ' + str(labels[i]) + '\n')

def get_IDs(line): 
    contents = line.strip().split('\t') 
    ID = contents[0]
    return ID

def find_semeval2013_dups():
    """
    Some target words show up twice in the same 
    context. We want to know which instances have
    this issue.
    """
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    data = sc.textFile(SEMEVAL_TEST_VECTORS)
    data = data.filter(semeval_words_of_interest)
    data = data.map(get_IDs)
    IDs = Counter(data.collect())
    sc.stop()
    with open(LOGS + 'semeval2013_test_dups', 'w') as outfile:
        for i in IDs: 
            if IDs[i] > 1: 
                outfile.write(i + '\n')

def get_dup_mapping(): 
    """
    Figure out which word (first, second, etc) is the actual target
    """
    nlp = spacy.load("en_core_web_sm")
    dups = set()
    with open(LOGS + 'semeval2013_test_dups', 'r') as infile: 
       for line in infile: 
           dups.add(line.strip().split('_')[0])
    sem_eval_test = '../SemEval-2013-Task-13-test-data/contexts/xml-format/'
    dup_map = {}
    for f in os.listdir(sem_eval_test): 
        tree = ET.parse(sem_eval_test + f)
        root = tree.getroot()
        lemma = f.replace('.xml', '')
        for instance in root: 
            if instance.attrib['id'] in dups: 
                tokens = nlp(instance.text, disable=['parser', 'tagger', 'ner'])
                target = instance.attrib['token']
                order = -1
                for t in tokens:
                    if t.text == target: 
                        order += 1
                    if t.idx == int(instance.attrib['tokenStart']):
                        dup_map[instance.attrib['id']] = order
    with open(LOGS + 'semeval2013_dup_map.json', 'w') as outfile:
        json.dump(dup_map, outfile)

def filter_semeval2013_vecs():
    with open(LOGS + 'semeval2013_dup_map.json', 'r') as infile:
        dup_map = json.load(infile)
    outfile = open(SEMEVAL_TEST_VECTORS2, 'w') 
    times_seen = Counter() # zero indexed
    with open(SEMEVAL_TEST_VECTORS, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            ID = contents[0].split('_')[-3] 
            token = contents[0].split('_')[-1]
            if token != contents[1]: continue
            if ID in dup_map: 
                if times_seen[ID] == dup_map[ID]: 
                    outfile.write(line)
                times_seen[ID] += 1
            else:
                outfile.write(line)
    outfile.close()


def semeval2010_cluster_training(): 
    '''
    Input: training vectors
    note that one ID might have multiple instances of a word
    Output: cluster centroids
    '''
    pass

def evaluate_nmi(): 
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
    #find_semeval2013_dups()
    #get_dup_mapping()
    #filter_semeval2013_vecs()
    semeval_clusters(test=True)
    evaluate_nmi()
    evaluate_bcubed()

if __name__ == "__main__":
    main()
