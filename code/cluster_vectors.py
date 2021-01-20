from pyspark import SparkConf, SparkContext
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
#import bcubed
from collections import defaultdict, Counter
import json
import os
import xml.etree.ElementTree as ET
import re
import string
#import spacy
from functools import partial
from sklearn.decomposition import PCA
import random
from sklearn.metrics.pairwise import cosine_similarity
import os.path
from joblib import dump, load
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
import time
#from pyclustering.cluster.xmeans import xmeans
#from pyclustering.cluster.encoder import type_encoding, cluster_encoder
#from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

wnl = WordNetLemmatizer()

ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
LOGS = ROOT + 'logs/'
SEMEVAL2010_TRAIN_VECTORS = LOGS + 'semeval2010/semeval2010_train_bert2' 
SEMEVAL2010_TEST_VECTORS = LOGS + 'semeval2010/semeval2010_test_bert2' 
SEMEVAL2013_TRAIN_VECTORS = LOGS + 'semeval2013/semeval2013_train_bert2'
SEMEVAL2013_TEST_VECTORS = LOGS + 'semeval2013/semeval2013_test_bert2'
SEMEVAL2013_TEST_VECTORS2 = LOGS + 'semeval2013/semeval2013_test_bert3'

def semeval_words_of_interest(line): 
    contents = line.strip().split('\t')
    token = contents[0].split('_')[-1]
    return token == contents[1]

def semeval_lemmas_of_interest(line): 
    contents = line.strip().split('\t') 
    ID = contents[0].split('_')
    lemma = ID[-2].split('.')[0]
    pos = ID[-2].split('.')[1]
    return lemma == wnl.lemmatize(contents[1], pos)

def get_semeval_vector(line): 
    contents = line.strip().split('\t') 
    ID = contents[0]
    lemma = ID.split('_')[-2] 
    vector = np.array([[float(i) for i in contents[2].split()]])
    return (lemma, ([ID], vector))

def xmeans_helper(tup, dim_reduct=None, semeval2010=False, rs=0, normalize=False): 
    lemma = tup[0]
    IDs = tup[1][0]
    data = tup[1][1]
    if dim_reduct is not None:
        if normalize: 
            outpath = LOGS + 'pca/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '_normalize.joblib' 
            scaler_path = LOGS + 'standardscaler/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '_normalize.joblib' 
            scaler = StandardScaler()
            data = scaler.fit_transform(data) 
            dump(scaler, scaler_path)
        else:  
            outpath = LOGS + 'pca/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '.joblib'
        pca = PCA(n_components=dim_reduct, random_state=rs)
        data = pca.fit_transform(data)
        dump(pca, outpath)
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(data, amount_initial_centers).initialize()
    xmeans_instance = xmeans(data, initial_centers, 20)
    xmeans_instance.process()
    encoding = xmeans_instance.get_cluster_encoding()
    clusters = xmeans_instance.get_clusters()
    encoder = cluster_encoder(encoding, clusters, data)
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING);
    clusters = encoder.get_clusters()
    centroids = xmeans_instance.get_centers()
    return (IDs, (clusters, centroids))

def kmeans_with_crit(tup, dim_reduct=None, semeval2010=False, rs=0, lamb=10000, normalize=False): 
    #start = time.time() # comment out when not timing
    lemma = tup[0]
    IDs = tup[1][0]
    data = tup[1][1]
    if dim_reduct is not None:
        if normalize: 
            outpath = LOGS + 'pca/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '_' + str(lamb) + '_normalize.joblib' 
            scaler_path = LOGS + 'standardscaler/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '_' + str(lamb) + '_normalize.joblib' 
            scaler = StandardScaler()
            data = scaler.fit_transform(data) 
            dump(scaler, scaler_path)
        else:  
            outpath = LOGS + 'pca/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '_' + str(lamb) + '.joblib'
        pca = PCA(n_components=dim_reduct, random_state=rs)
        data = pca.fit_transform(data)
        dump(pca, outpath)
    ks = range(2, 10)
    labels = {} # k : km.labels_
    centroids = {} 
    rss = np.zeros(len(ks))
    for i, k in enumerate(ks):
        km = KMeans(k, n_jobs=-1, random_state=rs)
        km.fit(data)
        rss[i] = km.inertia_
        labels[k] = km.labels_
        centroids[k] = km.cluster_centers_
    crits = []
    for i in range(len(ks)): 
        k = ks[i] 
        crit = rss[i]  + lamb*k
        crits.append(crit)
    best_k = np.argmin(crits)
    #end = time.time() # comment out when not timing
    #with open(LOGS + 'semeval2010_timing/' + lemma, 'w') as outfile: # comment out
    #    outfile.write(str(end-start)) # comment out
    return (IDs, (labels[ks[best_k]], centroids[ks[best_k]]))


def kmeans_with_gap_statistic(tup, dim_reduct=None, semeval2010=False, rs=0, normalize=False): 
    """
    Based off of https://anaconda.org/milesgranger/gap-statistic/notebook
    """
    lemma = tup[0]
    IDs = tup[1][0]
    data = tup[1][1]
    if dim_reduct is not None:
        if normalize: 
            outpath = LOGS + 'pca/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '_normalize.joblib' 
            scaler_path = LOGS + 'standardscaler/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '_normalize.joblib' 
            scaler = StandardScaler()
            data = scaler.fit_transform(data) 
            dump(scaler, scaler_path)
        else:  
            outpath = LOGS + 'pca/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '.joblib'
        pca = PCA(n_components=dim_reduct, random_state=rs)
        data = pca.fit_transform(data)
        dump(pca, outpath)
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
            km = KMeans(k, n_jobs=-1, random_state=rs)
            km.fit(random_ref)
            ref_disps[j] = km.inertia_
        km = KMeans(k, n_jobs=-1, random_state=rs)
        km.fit(data)
        orig_disp = km.inertia_
        l = np.mean(np.log(ref_disps))
        gap = l - np.log(orig_disp)
        s[i] = math.sqrt(1.0 + 1.0/nrefs)*np.std(np.log(ref_disps))
        gaps[i] = gap
        labels[k] = km.labels_
        centroids[k] = km.cluster_centers_
    for i in range(len(ks) - 1): 
        k = ks[i] 
        #if k == 4: return (IDs, (labels[k], centroids[k])) 
        if gaps[i] >= gaps[i+1] - s[i+1]:
            return (IDs, (labels[k], centroids[k]))
    return (IDs, (labels[ks[-1]], centroids[ks[-1]]))


def get_data_size(tup): 
    token = tup[0]
    data = tup[1][1]
    return (token, data.shape)
    
def semeval_clusters(test=False, dim_reduct=None): 
    """
    Note: we are not using this anymore. This function clusters
    the test set on its own without a training set determining
    the centroids beforehand. 
    """
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    if test: 
        data = sc.textFile(SEMEVAL2013_TEST_VECTORS2) 
    else: 
        data = sc.textFile(SEMEVAL2013_TRAIN_VECTORS)
    data = data.map(get_semeval_vector)
    data = data.reduceByKey(lambda n1, n2: (n1[0] + n2[0], np.concatenate((n1[1], n2[1]), axis=0)))
    #size = data.map(get_data_size).collectAsMap()
    data = data.map(partial(kmeans_with_gap_statistic, dim_reduct=dim_reduct, semeval2010=False))
    clustered_IDs = data.collect()
    sc.stop()
    #for token in size: 
    #    print(token, size[token])
    if test: 
        if dim_reduct is not None: 
            outname = 'semeval_test_clusters' + str(dim_reduct)
        else: 
            outname = 'semeval_test_clusters'
    else: 
        if dim_reduct is not None: 
            outname = 'semeval_clusters' + str(dim_reduct)
        else: 
            outname = 'semeval_clusters'
    with open(LOGS + outname, 'w') as outfile: 
        for tup in clustered_IDs: 
            IDs = tup[0]
            labels = tup[1][0]
            for i, ID in enumerate(IDs): 
                small_id = ID.split('_')[-3]
                lemma = ID.split('_')[-2]
                outfile.write(lemma + ' ' + small_id + ' ' + lemma + str(labels[i]) + '\n')

def get_IDs(line): 
    contents = line.strip().split('\t') 
    ID = contents[0]
    return ID

def find_semeval_dups(semeval2010=False):
    """
    Some target words show up twice in the same 
    context. We want to know which instances have
    this issue.
    """
    if semeval2010: 
        inpath = SEMEVAL2010_TEST_VECTORS
        outpath = 'semeval2010_test_dups'
    else: 
        inpath = SEMEVAL2013_TEST_VECTORS
        outpath = 'semeval2013_test_dups'
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    data = sc.textFile(inpath)
    data = data.filter(semeval_words_of_interest)
    data = data.map(get_IDs)
    IDs = Counter(data.collect())
    sc.stop()
    with open(LOGS + outpath, 'w') as outfile:
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

def sample_vectors(tup): 
    IDs = tup[1][0]
    X = np.array(tup[1][1])
    cutoff = 500
    if len(IDs) < cutoff: 
        return tup
    else: 
        idx = sorted(random.sample(range(len(IDs)), cutoff))
        IDs_sample = []
        for i in idx: 
            IDs_sample.append(IDs[i])
    return (tup[0], (IDs_sample, X[idx,:]))

def semeval_cluster_training(semeval2010=False, dim_reduct=None, rs=0, lamb=10000, normalize=False): 
    '''
    Input: training vectors
    note that one ID might have multiple instances of a word
    Output: cluster centroids
    This is how we are figuring out what our centroids are. 
    '''
    random.seed(rs)
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    if semeval2010: 
        outname = 'semeval2010/semeval2010_centroids/' 
        data = sc.textFile(SEMEVAL2010_TRAIN_VECTORS)
        #data = sc.textFile(SEMEVAL2010_TEST_VECTORS) # cluster directly on test set 
    else: 
        outname = 'semeval2013/semeval2013_centroids/'
        data = sc.textFile(SEMEVAL2013_TRAIN_VECTORS)
        #data = sc.textFile(SEMEVAL2013_TEST_VECTORS2)
    if semeval2010: 
        data = data.filter(semeval_lemmas_of_interest) 
    else: 
        data = data.filter(semeval_words_of_interest)
    data = data.map(get_semeval_vector)
    data = data.reduceByKey(lambda n1, n2: (n1[0] + n2[0], np.concatenate((n1[1], n2[1]), axis=0)))
    data = data.map(sample_vectors) 
    out = data.map(partial(kmeans_with_crit, dim_reduct=dim_reduct, 
         semeval2010=semeval2010, rs=rs, lamb=lamb, normalize=normalize)) 
    data.unpersist()
    data = None
    clustered_IDs = out.collect()
    sc.stop() 
    for tup in clustered_IDs: 
        ID = tup[0][0]
        lemma = ID.split('_')[-2]
        centroids = np.array(tup[1][1]) 
        if normalize: 
            np.save(LOGS + outname + lemma + '_' + str(dim_reduct) + '_' + str(rs) + 
                '_' + str(lamb) + '_normalized.npy', centroids)
        else: 
            np.save(LOGS + outname + lemma + '_' + str(dim_reduct) + '_' + str(rs) + \
                '_' + str(lamb) + '.npy', centroids)

def semeval_match_centroids(tup, semeval2010=False, dim_reduct=None, rs=0, lamb=10000, normalize=False):
    lemma = tup[0]
    IDs = tup[1][0]
    data = np.array(tup[1][1])
    if not semeval2010: 
        lemma = lemma.replace('.j', '.a') # semeval2013 train set has different letter for adj 
    if dim_reduct is not None:
        if normalize: 
            inpath = LOGS + 'pca/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '_' + str(lamb) + '_normalize.joblib' 
            scaler_path = LOGS + 'standardscaler/' + str(semeval2010) + '_' + lemma + '_' + \
                str(dim_reduct) + '_' + str(rs) + '_' + str(lamb) + '_normalize.joblib' 
            scaler = load(scaler_path)
            pca = load(inpath)
            data = scaler.transform(data)
            data = pca.transform(data)
        else: 
            inpath = LOGS + 'pca/' + str(semeval2010) + '_' + lemma + \
             '_' + str(dim_reduct) + '_' + str(rs) + '_' + str(lamb) + '.joblib'
            pca = load(inpath)
            data = pca.transform(data)
    if semeval2010: 
        inname = LOGS + 'semeval2010/semeval2010_centroids/' 
    else: 
        inname = LOGS + 'semeval2013/semeval2013_centroids/'
    if normalize: 
        centroids = np.load(inname + lemma + '_' + \
             str(dim_reduct) + '_' + str(rs) + '_' + str(lamb) + '_normalized.npy')
    else: 
        centroids = np.load(inname + lemma + '_' + \
             str(dim_reduct) + '_' + str(rs) + '_' + str(lamb) + '.npy')
    assert data.shape[1] == centroids.shape[1]
    sims = cosine_similarity(data, centroids) # IDs x n_centroids
    labels = np.argmax(sims, axis=1)
    ret = []
    for i in range(len(IDs)): 
        ret.append((IDs[i], labels[i]))
    return ret

def semeval_cluster_test(semeval2010=False, dim_reduct=None, rs=0, lamb=10000, normalize=False): 
    conf = SparkConf()
    sc = SparkContext(conf=conf) 
    if semeval2010: 
        outname = 'semeval2010/semeval2010_clusters' + str(dim_reduct) + '_' + str(rs) + '_' + str(lamb)
        data = sc.textFile(SEMEVAL2010_TEST_VECTORS)
    else: 
        outname = 'semeval2013/semeval2013_clusters' + str(dim_reduct) + '_' + str(rs) + '_' + str(lamb)
        data = sc.textFile(SEMEVAL2013_TEST_VECTORS2)
    if normalize: 
        outname += "_normalize"
    if semeval2010: 
        data = data.filter(semeval_lemmas_of_interest) 
    else: 
        data = data.filter(semeval_words_of_interest)
    data = data.map(get_semeval_vector)
    data = data.reduceByKey(lambda n1, n2: (n1[0] + n2[0], np.concatenate((n1[1], n2[1]), axis=0)))
    out = data.flatMap(partial(semeval_match_centroids, semeval2010=semeval2010, 
        dim_reduct=dim_reduct, rs=rs, lamb=lamb, normalize=normalize))
    data.unpersist()
    data = None
    id_labels = out.collectAsMap()
    sc.stop()
    with open(LOGS + outname, 'w') as outfile: 
        for ID in id_labels: 
            label = id_labels[ID]
            small_id = ID.split('_')[-3]
            lemma = ID.split('_')[-2]
            outfile.write(lemma + ' ' + small_id + ' ' + lemma + str(label) + '\n')

def read_labels_for_eval(goldpath, mypath): 
    gold_labels = defaultdict(dict)
    my_labels = defaultdict(dict)
    gold_c = defaultdict(list)
    my_c = defaultdict(list)
    labels = []
    gold_lemma_labels = defaultdict(set)
    my_lemma_labels = defaultdict(set)
    with open(goldpath, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split()
            lemma = contents[0]
            instance = contents[1]
            label = contents[2]
            if label not in labels: 
                labels.append(label)
            i = labels.index(label)
            gold_labels[lemma][instance] = set([i])
            gold_lemma_labels[lemma].add(label)
            gold_c[i].append(instance)
    labels = []  
    with open(mypath, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split()
            lemma = contents[0]
            instance = contents[1]
            label = contents[2] 
            if instance not in gold_labels[lemma]: continue # only evaluate on things in gold
            if label not in labels: 
                labels.append(label)
            i = labels.index(label)
            my_labels[lemma][instance] = set([i])
            my_lemma_labels[lemma].add(label)
            my_c[label].append(instance) 
    print("num gold clusters", len(gold_c), "num my clusters", len(my_c))
    for lemma in gold_lemma_labels: 
        print(lemma, len(gold_lemma_labels[lemma]), len(my_lemma_labels[lemma]))
    return gold_labels, my_labels

def count_centroids(dim_reduct=100, rs=0): 
    for f in os.listdir(LOGS + 'semeval2010/semeval2010_centroids/'):
        if f.endswith('_' + str(dim_reduct) + '_' + str(rs) + '.npy'): 
            centroids = np.load(LOGS + 'semeval2010/semeval2010_centroids/' + f) 
            lemma = f.split('_')[0] 
            print(lemma, centroids.shape[0])
 
def main(): 
    #find_semeval_dups(semeval2010=True)
    #get_dup_mapping()
    #filter_semeval2013_vecs()
    #semeval_clusters(test=True, dim_reduct=20)
    #for dr in [150]:
    #    for lamb in [5000]:    
    #        for r in range(2, 5): 
    #for r in range(5): 
    #    for lamb in [10000, 15000, 5000, 1000, 20000]: 
    for r in range(2, 5):
         lamb = 10000
         semeval_cluster_training(semeval2010=False, dim_reduct=None, rs=r, lamb=lamb)
         semeval_cluster_test(semeval2010=False, dim_reduct=None, rs=r, lamb=lamb)
    
    #read_labels_for_eval('../semeval-2010-task-14/evaluation/unsup_eval/keys/all.key', 
    #    LOGS + 'semeval2010/semeval2010_clusters100_1')

if __name__ == "__main__":
    main()
