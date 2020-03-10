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
root = '/data0/lucy/ingroup_lang/'

def get_values(path, feature_dict): 
    with open(path, 'r') as infile: 
        for line in infile: 
            if line.startswith('subreddit,'): continue
            contents = line.strip().split(',')
            feature_dict[contents[0].lower()].append(float(contents[1]))
    return feature_dict

def get_data(sociolect_metric): 
    feature_dict = defaultdict(list) # subreddit_name : [features]
    feature_names = []
    size_path = root + 'logs/commentor_counts/part-00000-64b1d705-9cf8-4a54-9c4d-598e5bf9085f-c000.csv'
    feature_names.append('community size')
    feature_dict = get_values(size_path, feature_dict)
    activity_path = root + 'logs/commentor_activity'
    feature_names.append('user activity')
    feature_dict = get_values(activity_path, feature_dict)
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]: 
        loyalty_path = root + 'logs/commentor_loyalty_'+str(int(threshold*100))
        feature_names.append('user loyalty ' + str(int(threshold*100)))
        feature_dict = get_values(loyalty_path, feature_dict)
    commentor_path = root + 'logs/commentor_density'
    feature_names.append('commentor density')
    feature_dict = get_values(commentor_path, feature_dict)
    X = []
    y = []
    y_bin = []
    count_cut_off = 0
    if sociolect_metric == 'pmi': 
        path = root + '/logs/pmi/'
        cut_off = 0.2
    elif sociolect_metric == 'tfidf': 
        path = root + '/logs/tfidf/'
        cut_off = 2
    elif sociolect_metric == 'max_pmi': 
        path = root + 'logs/ft_max_sense_pmi/'
        cut_off = 0.2
    for sr in sorted(feature_dict.keys()): 
        assert len(feature_dict[sr]) == 8
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

def predict_ols(sociolect_metric): 
    X, y, y_bin, feature_names = get_data(sociolect_metric)
    model = sm.OLS(y, X)
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
    
def u_tests(sociolect_metric): 
    '''
    less sociolect-y communities are larger, less active, less loyal, less dense
    '''
    print(sociolect_metric)
    X, y, y_bin, feature_names = get_data(sociolect_metric)
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
    #u_tests('pmi')
    #u_tests('tfidf')
    u_tests('max_pmi')
    #matching_subreddits('community size', ['user activity', 'user loyalty 50', 'commentor density'], 'pmi')

if __name__ == "__main__":
    main()
