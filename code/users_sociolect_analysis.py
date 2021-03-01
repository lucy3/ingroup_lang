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

def get_features(include_topics=False, factor_topics=False, include_subs=False): 
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
    
    if include_subs: 
        sub_ratio_path = root + 'logs/subscribers_ratio'
        feature_names.append('subscriber ratio')
        feature_dict = get_values(sub_ratio_path, feature_dict)
        sub_path = root + 'logs/subscribers'
        feature_names.append('subscribers')
        feature_dict = get_values(sub_path, feature_dict)
    
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

def get_data(sense_cutoff, type_cutoff, include_topics=False, factor_topics=False, ag=False, include_subs=False):
    feature_dict, feature_names = get_features(include_topics=include_topics, 
                                               factor_topics=factor_topics, 
                                               include_subs=include_subs)
    X = []
    y = []
    y_bin = []
    type_path = root + '/logs/norm_pmi/'
    if ag: 
        sense_path = root + 'logs/ag_most_sense_pmi/'
    else:
        sense_path = root + 'logs/base_most_sense_pmi/'
    for sr in tqdm.tqdm(sorted(feature_dict.keys())): 
        X.append(feature_dict[sr])
        sense_scores = defaultdict(float)
        vocab_size = 0
        sociolect_count = 0
        with open(sense_path + sr + '.csv', 'r') as infile:
            reader = csv.DictReader(infile)
            for row in reader: 
                w = row['word']
                score = float(row['most_pmi'])
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

def get_data_old(sociolect_metric, cut_off, include_subs=False): 
    '''
    Returns: 
        X - each column is a user feature, each row is an example
        y - sociolect-y score, or fraction of terms above a cut off
        y_bin - 
        feature_names
    '''
    feature_dict, feature_names = get_features(include_subs=include_subs)
    X = []
    y = []
    y_bin = []
    count_cut_off = 0
    if sociolect_metric == 'pmi': 
        path = root + '/logs/norm_pmi/'
    elif sociolect_metric == 'tfidf': 
        path = root + '/logs/tfidf/'
    elif sociolect_metric == 'ag_most_pmi': 
        path = root + 'logs/ag_most_sense_pmi/'
        sociolect_metric = 'most_pmi'
    elif sociolect_metric == 'base_most_pmi': 
        path = root + 'logs/base_most_sense_pmi/'
        sociolect_metric = 'most_pmi'
    for sr in sorted(feature_dict.keys()): 
        X.append(feature_dict[sr])
        f = sr + '_0.2.csv' 
        if sociolect_metric.endswith('most_pmi'): 
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

def predict_ols(include_topics=False): 
    sense_cutoff = 0.17994827737953864
    type_cutoff = 0.3034868476499491
    X, y, y_bin, feature_names = get_data(sense_cutoff, type_cutoff, ag=False, include_topics=include_topics, 
                                          factor_topics=False, include_subs=False)
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
    base_scs = [0.0172226043675457, 0.029333511361664382, 0.04439566287200495, 0.06524046644385789, 0.10001508696572367, 0.17994827737953864, 0.2144383944959922]
    tcs = [0.008950423272301661, 0.02261513150464957, 0.04037423905310003, 0.06720029386461164, 0.12191657724645635, 0.3034868476499491, 0.3517743621999161]
    #tcs = [5.006313171329149] 
    #ag_scs = [2.66291875427572]
    #base_scs = [2.8615285178167453]
    if sociolect_metric is None: 
        for i in range(len(tcs)): 
            print("Combined type and sense U-tests")
            sense_cutoff = base_scs[i]
            type_cutoff = tcs[i]
            X, y, y_bin, feature_names = get_data(sense_cutoff, type_cutoff, ag=False, include_subs=True)
            run_u_test('community size', feature_names.index('community size'), X, y_bin, 'greater')
            run_u_test('user activity', feature_names.index('user activity'), X, y_bin, 'less')
            run_u_test('user loyalty 50', feature_names.index('user loyalty 50'), X, y_bin, 'less')
            run_u_test('commentor density', feature_names.index('commentor density'), X, y_bin, 'less')
            run_u_test('subscriber ratio', feature_names.index('subscriber ratio'), X, y_bin, 'greater')
            run_u_test('subscribers', feature_names.index('subscribers'), X, y_bin, 'greater')
    else:
        print(sociolect_metric)
        if sociolect_metric == 'pmi':
            cut_offs = tcs
        elif sociolect_metric == 'ag_most_pmi':
            cut_offs = ag_scs
        elif sociolect_metric == 'base_most_pmi':
            cut_offs = base_scs
        for i in range(len(cut_offs)): 
            X, y, y_bin, feature_names = get_data_old(sociolect_metric, cut_offs[i], include_subs=True)
            run_u_test('community size', feature_names.index('community size'), X, y_bin, 'greater')
            run_u_test('user activity', feature_names.index('user activity'), X, y_bin, 'less')
            run_u_test('user loyalty 50', feature_names.index('user loyalty 50'), X, y_bin, 'less')
            run_u_test('commentor density', feature_names.index('commentor density'), X, y_bin, 'less')
            run_u_test('subscriber ratio', feature_names.index('subscriber ratio'), X, y_bin, 'greater')
            run_u_test('subscribers', feature_names.index('subscribers'), X, y_bin, 'greater')
    
def main(): 
    #u_tests('pmi')
    #u_tests('base_most_pmi')
    #u_tests()

    predict_ols()

if __name__ == "__main__":
    main()
