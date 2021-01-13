from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering
from pyspark import SparkConf, SparkContext
import numpy as np
from collections import defaultdict, Counter
import time
import random
from functools import partial
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
LOGS = ROOT + 'logs/'
SEMEVAL2010_TRAIN_VECTORS = LOGS + 'semeval2010/semeval2010_train_bert2' 
SEMEVAL2010_TEST_VECTORS = LOGS + 'semeval2010/semeval2010_test_bert2' 
SEMEVAL2013_TRAIN_VECTORS = LOGS + 'semeval2013/semeval2013_train_bert2'
SEMEVAL2013_TEST_VECTORS = LOGS + 'semeval2013/semeval2013_test_bert2'
SEMEVAL2013_TEST_VECTORS2 = LOGS + 'semeval2013/semeval2013_test_bert3'

wnl = WordNetLemmatizer()
np.seterr(all='raise')

def spectral_cluster(tup, semeval2010=False, rs=0): 
    lemma = tup[0]
    IDs = tup[1][0]
    data = tup[1][1]
    neighbor_rank = 7 # from local scaling paper
    d = euclidean_distances(data)
    A = np.square(d)
    omegas = np.argsort(d)[:,neighbor_rank-1] # first index is smallest
    omegas = d[np.arange(d.shape[0]),omegas] # d(x, x_7)
    zeros = np.where(omegas == 0)[0]
    if len(zeros) > 0: 
        print("---------------- ZEROS", omegas)
        print("-------------------------")
        print(d[np.where(omegas==0), :])
    A = -1*A
    A = A / omegas[:,None]
    A = A / omegas[None,:]
    A = np.exp(A) # affinity matrix
    # compute Laplacian
    L = csgraph.laplacian(A, normed=True)
    # get smallest 10 eigenvalues of L, sorted by value smallest to largest
    w, v = np.linalg.eig(L)
    ev = sorted(w)[:10]
    print("************* eigenvalues", lemma, ev)
    gaps = []
    for i in range(len(ev)-1): 
        gaps.append(ev[i+1] - ev[i])
    # eigengap
    k = np.argmax(gaps)
    if k <= 1: k = 2 # set a lower bound
    clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=0).fit(A)
    labels = clustering.labels_
    return (IDs, labels, data)

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

def semeval_cluster_training(semeval2010=True, rs=0): 
    random.seed(rs)
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    if semeval2010: 
        outname = LOGS + 'spectral_results/semeval2010/' 
        data = sc.textFile(SEMEVAL2010_TRAIN_VECTORS)
    else: 
        outname = LOGS + 'spectral_results/semeval2013/' 
        data = sc.textFile(SEMEVAL2013_TRAIN_VECTORS)
    if semeval2010: 
        data = data.filter(semeval_lemmas_of_interest) 
    else: 
        data = data.filter(semeval_words_of_interest)
    data = data.map(get_semeval_vector)
    data = data.reduceByKey(lambda n1, n2: (n1[0] + n2[0], np.concatenate((n1[1], n2[1]), axis=0)))
    data = data.map(sample_vectors) 
    out = data.map(partial(spectral_cluster, 
         semeval2010=semeval2010, rs=rs)) 
    labels = out.collect()
    sc.stop() 
    for tup in labels: 
        labels = tup[1]
        lemma = tup[0][0].split('_')[-2]
        data = tup[2]
        prefix = outname + lemma + '_' + str(rs)
        with open(prefix + '_trainlabels.txt', 'w') as outfile: 
            for label in labels: 
                outfile.write(str(label) + '\n')
        np.save(prefix + '_traindata.npy', data)
        
def semeval_match_nn(tup, semeval2010=True, rs=0): 
    lemma = tup[0]
    IDs = tup[1][0]
    data = np.array(tup[1][1])
    # TODO: load up train_labels based on lemma or ID
    # TODO: load up training examples
    sim = cosine_similarity(data, train_data)
    max_idx = np.argmax(sim, axis=1)
    labels = [train_labels[i] for i in max_idx]  
    ret = []
    for i in range(len(IDs)): 
        ret.append((IDs[i], labels[i]))
    return ret

def semeval_cluster_test(semeval2010=True, rs=0): 
    conf = SparkConf()
    sc = SparkContext(conf=conf) 
    if semeval2010: 
        outname = LOGS + 'spectral_results/semeval2010/semeval2010_clusters_' + str(rs) 
        data = sc.textFile(SEMEVAL2010_TEST_VECTORS)
    else: 
        outname = LOGS + 'spectral_results/semeval2013/semeval2013_clusters_' + str(rs)  
        data = sc.textFile(SEMEVAL2013_TEST_VECTORS2)
    if semeval2010: 
        data = data.filter(semeval_lemmas_of_interest) 
    else: 
        data = data.filter(semeval_words_of_interest)
    data = data.map(get_semeval_vector)
    data = data.reduceByKey(lambda n1, n2: (n1[0] + n2[0], np.concatenate((n1[1], n2[1]), axis=0)))
    out = data.flatMap(partial(semeval_match_nn, semeval2010=semeval2010, rs=rs))
    data.unpersist()
    data = None
    id_labels = out.collectAsMap()
    sc.stop()
    with open(outname, 'w') as outfile: 
        for ID in id_labels: 
            label = id_labels[ID]
            small_id = ID.split('_')[-3]
            lemma = ID.split('_')[-2]
            outfile.write(lemma + ' ' + small_id + ' ' + lemma + str(label) + '\n')

def main(): 
    #for r in range(5):
    r = 0
    semeval_cluster_training(semeval2010=True, rs=r)
    #semeval_cluster_test(semeval2010=True, rs=r)

if __name__ == "__main__":
    main()
