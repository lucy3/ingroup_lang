from pyspark import SparkConf, SparkContext
import numpy as np
import math
from sklearn.cluster import KMeans
import numpy as np

conf = SparkConf()
sc = SparkContext(conf=conf)

LOGS = '/global/scratch/lucy3_li/ingroup_lang/logs/'
SEMEVAL_VECTORS = LOGS + 'semeval2013_bert'

def semeval_words_of_interest(line): 
    contents = line.strip().split('\t')
    token = contents[0].split('_')[-1]
    return token == contents[1]

def get_semeval_vector(line): 
    contents = line.strip().split('\t') 
    ID = contents[0]
    token = ID.split('_')[-1]
    vector = np.array(contents[2].split())
    return (token, ([ID], [vector]))

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
    s = []
    for i, k in enumerate(ks):
        ref_disps = np.zeros(nrefs)
        for j in range(nrefs):
            random_ref = np.random.random_sample(size=data.shape)
            km = KMeans(k)
            km.fit(random_ref)
            ref_disps[j] = km.inertia_
        km = KMeans(k)
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
    
def semeval_clusters(): 
    data = sc.textFile(SEMEVAL_VECTORS) 
    data = data.filter(semeval_words_of_interest)
    data = data.map(get_semeval_vector)
    data = data.reduceByKey(lambda n1, n2: (n1[0] + n2[0], n1[1] + n2[1])) 
    data = data.map(lambda x: (x[0], np.array(x[1])))
    data = data.map(kmeans_with_gap_statistic)
    clustered_IDs = data.collect()
    sc.stop()
    with open(LOGS + 'semeval_clusters', 'w') as outfile: 
        for tup in clustered_IDs: 
            IDs = tup[0]
            labels = tup[1]
            for i, ID in enumerate(IDs): 
                outfile.write(IDs[i] + '\t' + str(labels[i]) + '\n')

def main(): 
    semeval_clusters()

if __name__ == "__main__":
    main()
