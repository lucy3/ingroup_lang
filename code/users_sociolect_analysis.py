"""
This file has functions for: 
- figuring out if pmi or tfidf better aligns with subreddit glossaries
- which user metric is most predictive of in-group language? 
- controlling for topic in looking at sociolects
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from collections import defaultdict

root = '/data0/lucy/ingroup_lang/'

def get_values(path, feature_dict): 
    with open(path, 'r') as infile: 
        for line in infile: 
            if line.startswith('subreddit,'): continue
            contents = line.strip().split(',')
            feature_dict[contents[0].lower()].append(float(contents[1]))
    return feature_dict

def predict_sociolects(sociolect_metric='pmi_words'):
    """
    X is [community size, user activity, user loyalty, user density] 
    where each row is a subreddit
    y is # of high pmi words, # of high tf-idf words, or # of high pmi senses
    """ 
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
    for sr in sorted(feature_dict.keys()): 
        assert len(feature_dict[sr]) == 8
        X.append(feature_dict[sr])
    X = np.array(X)
    y = []
    # todo: get y 
    clf1 = LinearRegression(n_jobs=-1)
    clf1.fit(X, y)
    print("Weights for features:")
    for i, cf in enumerate(feature_names): 
        print(cf, clf1.coef_[i])
    clf2 = LinearRegression(n_jobs=-1) 
    scores = cross_val_score(clf2, X, y, cv=5)
    print("5-fold cv score mean:", np.mean(scores))

def main(): 
    predict_sociolects()

if __name__ == "__main__":
    main()
