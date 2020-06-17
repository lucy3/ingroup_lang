"""
This file has functions for: 
- figuring out if pmi or tfidf better aligns with subreddit glossaries
- which user metric is most predictive of in-group language? 
- controlling for topic in looking at sociolects
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import os
import pandas as pd
from scipy.stats import zscore 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
import math
import random
import csv
import json
import tqdm
root = '/mnt/data0/lucy/ingroup_lang/'

def get_values(path, feature_dict): 
    with open(path, 'r') as infile: 
        for line in infile: 
            if line.startswith('subreddit,'): continue
            contents = line.strip().split(',')
            feature_dict[contents[0].lower()].append(float(contents[1]))
    return feature_dict

def get_features(include_topics=False, factor_topics=False): 
    feature_dict = defaultdict(list) # subreddit_name : [features]
    feature_names = []
    size_path = root + 'logs/commentor_counts/part-00000-64b1d705-9cf8-4a54-9c4d-598e5bf9085f-c000.csv'
    feature_names.append('community size')
    feature_dict = get_values(size_path, feature_dict)
    activity_path = root + 'logs/commentor_activity'
    feature_names.append('user activity')
    feature_dict = get_values(activity_path, feature_dict)
    for threshold in [0.5]: 
        loyalty_path = root + 'logs/commentor_loyalty_'+str(int(threshold*100))
        feature_names.append('user loyalty ' + str(int(threshold*100)))
        feature_dict = get_values(loyalty_path, feature_dict)
    commentor_path = root + 'logs/commentor_density'
    feature_names.append('commentor density')
    feature_dict = get_values(commentor_path, feature_dict)
    topic_list = ['Discussion', 'Humor', 'Entertainment_Sports', 
                  'Lifestyle_Technology', 'Lifestyle_Relationships/Sex', 
                  'Other_Geography', 'Entertainment_Internet/Apps', 'Other_Cringe', 
                  'Other_Disgusting/Angering/Scary/Weird', 'Entertainment_TV', 
                  'Entertainment_Video games', 'Hobbies/Occupations']
    if include_topics: 
        with open(root + 'logs/topic_assignments.json', 'r') as infile: 
            topic_assignments = json.load(infile)
        for sr in topic_assignments: 
            if factor_topics: 
                topic_feats = [0 for i in range(len(topic_list))]
                if topic_assignments[sr] != 'Other': 
                    topic_feats[topic_list.index(topic_assignments[sr])] = 1
                feature_dict[sr].extend(topic_feats)
            else: 
                if topic_assignments[sr] in set(['Hobbies/Occupations', 'Entertainment_TV', 
                                                 'Lifestyle_Technology', 'Entertainment_Sports', 
                                                 'Entertainment_Video games', 'Other']): 
                    feature_dict[sr].append(1)
                else: 
                    feature_dict[sr].append(0)
        feature_names.append('topic')
    return feature_dict, feature_names

def get_data(sense_cutoff, type_cutoff, include_topics=False, factor_topics=False):
    feature_dict, feature_names = get_features(include_topics=include_topics, 
                                               factor_topics=factor_topics)
    X = []
    y = []
    y_bin = []
    type_path = root + '/logs/pmi/'
    sense_path = root + 'logs/ft_max_sense_pmi/'
    for sr in tqdm.tqdm(sorted(feature_dict.keys())): 
        X.append(feature_dict[sr])
        sense_scores = defaultdict(float)
        vocab_size = 0
        sociolect_count = 0
        with open(sense_path + sr + '.csv', 'r') as infile:
            reader = csv.DictReader(infile)
            for row in reader: 
                w = row['word']
                score = float(row['max_pmi'])
                sense_scores[w] = score
        with open(type_path + sr + '_0.2.csv', 'r') as infile:
            reader = csv.DictReader(infile)
            for row in reader: 
                w = row['word']
                score = float(row['pmi'])
                vocab_size += 1
                if score > type_cutoff or sense_scores[w] > sense_cutoff:  
                    sociolect_count += 1
        y.append(sociolect_count / float(vocab_size))
    median = np.median(y)
    for score in y: 
        y_bin.append(int(score >= median))
    X = np.array(X)
    y = np.array(y)
    y_bin = np.array(y_bin)
    assert X.shape[0] == y.shape[0]
    # z-score the columns
    X = zscore(X, axis=0)
    return X, y, y_bin, feature_names

def get_data_old(sociolect_metric, cut_off): 
    '''
    Returns: 
        X - each column is a user feature, each row is an example
        y - sociolect-y score, or fraction of terms above a cut off
        y_bin - 
        feature_names
    '''
    feature_dict, feature_names = get_features()
    X = []
    y = []
    y_bin = []
    count_cut_off = 0
    if sociolect_metric == 'pmi': 
        path = root + '/logs/pmi/'
    elif sociolect_metric == 'tfidf': 
        path = root + '/logs/tfidf/'
    elif sociolect_metric == 'max_pmi': 
        path = root + 'logs/ft_max_sense_pmi/'
    for sr in sorted(feature_dict.keys()): 
        X.append(feature_dict[sr])
        f = sr + '_0.2.csv' 
        if sociolect_metric == 'max_pmi': 
            f = sr + '.csv'
        df = pd.read_csv(path + f, engine='python')
        notable_words = df[df['count'] > count_cut_off]
        num_words = len(notable_words)
        high_val_df = notable_words[notable_words[sociolect_metric] > cut_off] 
        num_high_val = len(high_val_df)
        score = num_high_val / float(num_words)
        y.append(score)
    median = np.median(y)
    for score in y: 
        y_bin.append(int(score >= median))
    X = np.array(X)
    y = np.array(y)
    y_bin = np.array(y_bin)
    assert X.shape[0] == y.shape[0]
    # z-score the columns
    X = zscore(X, axis=0)
    return X, y, y_bin, feature_names

def predict_sociolects(sociolect_metric):
    """
    X is [community size, user activity, user loyalty, user density] 
    where each row is a subreddit
    y is # of high pmi words, # of high tf-idf words, or # of high pmi senses
    """ 
    X, y, y_bin, feature_names = get_data(sociolect_metric)
    clf1 = LinearRegression(n_jobs=-1)
    clf1.fit(X, y)
    print("LINEAR REGRESSION")
    print("Weights for features:")
    for i, cf in enumerate(feature_names): 
        print(cf, clf1.coef_[i])
    clf2 = LinearRegression(n_jobs=-1) 
    scores = cross_val_score(clf2, X, y, cv=3)
    print("3-fold cv R^2", scores)
    print("3-fold cv R^2 mean:", np.mean(scores))
    print
    clf7 = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
    clf7.fit(X, y)
    print("REGRESSION RF")
    print("Weights for features:")
    for i, cf in enumerate(feature_names): 
        print(cf, clf7.feature_importances_[i])
    clf8 = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
    scores = cross_val_score(clf8, X, y, cv=3)
    print("3-fold cv R^2", scores)
    print("3-fold cv R^2 mean:", np.mean(scores))
    print

    clf3 = LogisticRegression(solver='liblinear')
    clf3.fit(X, y_bin)
    print("LOGISTIC REGRESSION")
    print("Weights for features:")
    for i, cf in enumerate(feature_names): 
        print(cf, clf3.coef_[0,i])
    clf4 = LogisticRegression() 
    scores = cross_val_score(clf4, X, y_bin, cv=3, scoring='f1_macro')
    print("3-fold cv F1", scores)
    print("3-fold cv F1 mean:", np.mean(scores))
    print

    clf5 = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    clf5.fit(X, y_bin)
    print("BINARY CLASSIFIER RF")
    print("Weights for features:")
    for i, cf in enumerate(feature_names): 
        print(cf, clf5.feature_importances_[i])
    clf6 = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    scores = cross_val_score(clf6, X, y_bin, cv=3, scoring='f1_macro')
    print("3-fold cv F1", scores)
    print("3-fold cv F1 mean:", np.mean(scores))

    print("--------------------------")
    print

def predict_ols(sociolect_metric=None): 
    if sociolect_metric is None:
        sense_cutoff = 3.038046754473495
        type_cutoff = 5.008815935891599
        X, y, y_bin, feature_names = get_data(sense_cutoff, type_cutoff, include_topics=True, factor_topics=True)
    else:
        X, y, y_bin, feature_names = get_data(sociolect_metric)
    X_1 = sm.add_constant(X)
    model = sm.OLS(y, X_1)
    results = model.fit()
    print(feature_names)
    print(results.summary())
    
def run_u_test(name, col, X, y_bin, alternative): 
    '''
    first argument of mannwhitneyu is less sociolect-y 
    communities, second argument is more sociolect-y communities
    '''
    print(name)
    x = X[np.argwhere(y_bin == 0),col]
    y = X[np.argwhere(y_bin == 1),col]
    res = mannwhitneyu(x, y, alternative=alternative)
    print(res)
    
def u_tests(sociolect_metric=None): 
    '''
    less sociolect-y communities are larger, less active, less loyal, less dense
    '''
    # values copied from "senses" Python Notebook
    #scs = [0.5007301153953705, 0.7039233649614907, 0.953026733779678, 1.2910247870623466, 1.8434940980468622, 3.038046754473495, 3.520420472064922]
    #tcs = [0.143621134298, 0.3674494252902001, 0.6634994838245996, 1.1110732048100005, 2.0168260770319986, 5.008815935891599, 5.884734049733797]
    scs = [3.038046754473495] 
    tcs = [5.940024063721074]
    if sociolect_metric is None: 
        for i in range(len(scs)): 
            print("Combined type and sense U-tests")
            sense_cutoff = scs[i]
            type_cutoff = tcs[i]
            X, y, y_bin, feature_names = get_data(sense_cutoff, type_cutoff)
            run_u_test('community size', feature_names.index('community size'), X, y_bin, 'greater')
            run_u_test('user activity', feature_names.index('user activity'), X, y_bin, 'less')
            run_u_test('user loyalty 50', feature_names.index('user loyalty 50'), X, y_bin, 'less')
            run_u_test('commentor density', feature_names.index('commentor density'), X, y_bin, 'less')
    else:
        print(sociolect_metric)
        if sociolect_metric == 'pmi':
            cut_offs = tcs
        elif sociolect_metric == 'max_pmi':
            cut_offs = scs
        for i in range(len(cut_offs)): 
            X, y, y_bin, feature_names = get_data_old(sociolect_metric, cut_offs[i])
            run_u_test('community size', feature_names.index('community size'), X, y_bin, 'greater')
            run_u_test('user activity', feature_names.index('user activity'), X, y_bin, 'less')
            run_u_test('user loyalty 50', feature_names.index('user loyalty 50'), X, y_bin, 'less')
            run_u_test('commentor density', feature_names.index('commentor density'), X, y_bin, 'less')
    
def matching_subreddits(feature, matching_features, sociolect_metric): 
    '''
    Separate into "high" and "low" 
    '''
    print("Matching based on", matching_features, "to estimate effect of", \
          feature, "on", sociolect_metric)
    X, y, y_bin, feature_names = get_data(sociolect_metric)
    print("Number of examples:", X.shape[0])
    subset_features = matching_features + [feature]
    idx = []
    new_feature_names = []
    for i in range(len(feature_names)): 
        if feature_names[i] in subset_features: 
            idx.append(i)
            new_feature_names.append(feature_names[i])
    idx = np.array(idx)
    X = X[:,idx] # get only features we care about
    y = y.reshape((y.shape[0], 1))
    X = np.hstack((X, y)) # get y as well 
    feature_idx = new_feature_names.index(feature)
    vals = X[:,feature_idx]
    middle = np.median(vals) 
    # split into high and low values
    print("Splitting", feature, "into control and treatment based on high/low...")
    top_X = X[np.where(X[:,feature_idx] >= middle)]
    bottom_X = X[np.where(X[:,feature_idx] < middle)]
    top_buckets = defaultdict(list) # (feat1, feat2, ...) : [rows in that bucket]
    bottom_buckets = defaultdict(list) # (feat1, feat2, ...) : [rows in that bucket]
    #bucket_map = {'user activity': (lambda x: int(math.ceil(x / 10.0)) * 10), 
    #              'user loyalty 50': (lambda x: round(x, 1)),
    #              'commentor density': (lambda x: round(x, 2)),
    #              'community size': (lambda x: int(math.log10(x)))
    #        }
    for i in range(top_X.shape[0]): 
        arr = top_X[i]
        bucket = []
        for feat in sorted(matching_features): 
            v = round(arr[new_feature_names.index(feat)], 1)
            bucket.append(v)
        top_buckets[tuple(bucket)].append(arr)
    for i in range(bottom_X.shape[0]): 
        arr = bottom_X[i]
        bucket = []
        for feat in sorted(matching_features): 
            v = round(arr[new_feature_names.index(feat)], 1)
            bucket.append(v)
        bottom_buckets[tuple(bucket)].append(arr)
    # for each key in top, see if key exists in bottom and match as many as we can
    matches = []
    print("Matching buckets...")
    for key in top_buckets: 
        if key in bottom_buckets: 
            num_examples = min(len(top_buckets[key]), len(bottom_buckets[key]))
            new_values = random.sample(top_buckets[key], num_examples) + \
                random.sample(bottom_buckets[key], num_examples)
            if num_examples == 1: 
                print(new_values)
            matches.extend(new_values)
    new_X = np.array(matches)
    y = new_X[:,-1]
    new_X = new_X[:,:-1]
    print("Number of examples", new_X.shape[0])
    print("Running OLS...")
    model = sm.OLS(y, new_X)
    results = model.fit()
    print(new_feature_names)
    print(results.summary())

def main(): 
    #predict_sociolects('pmi')
    #predict_sociolects('tfidf')
    #predict_ols('pmi')
    #predict_ols('tfidf')
    u_tests('pmi')
    #u_tests('tfidf')
    #u_tests('max_pmi')
    #u_tests()
    #predict_ols()
    #matching_subreddits('community size', ['user activity', 'user loyalty 50', 'commentor density'], 'pmi')

if __name__ == "__main__":
    main()
