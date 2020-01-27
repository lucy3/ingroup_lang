"""
File for use with parallel where each word's 500
instances are passed through BERT and clustered

Functions are copied from
bert_vectors.py
cluster_vectors.py
"""
import os,sys,argparse
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import time
import xml.etree.ElementTree as ET
import math
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
from collections import defaultdict, Counter
import json
import xml.etree.ElementTree as ET
import re
import string
import spacy
from functools import partial
from sklearn.decomposition import PCA
import random
import os.path
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
LOGS = ROOT + 'logs/'

batch_size=32
dropout_rate=0.25
bert_dim=768


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('')
print("********************************************")
print("Running on: {}".format(device))
print("********************************************")
print('')

class EmbeddingClusterer():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.model.to(device)

    def read_instances(self, inputfile): 
        print("Reading sentences...")
        sentences = []
        i = 0
        line_number = 0
        with open(inputfile, 'r') as infile: 
            for line in infile: 
                contents = line.strip()
                sent_tok = sent_tokenize(contents)
                for sent in sent_tok: 
                    sentences.append((str(line_number), sent))
                line_number += 1
        return sentences

    def get_batches(self, sentences, max_batch): 
        print("Getting batches...")
        # each item in these lists is a sentence
        all_data = [] # indexed tokens, or word IDs
        all_words = [] # tokenized_text, or original words
        all_masks = [] 
        all_users = []
        for sentence in sentences: 
            marked_text = sentence[1] 
            tokenized_text = self.tokenizer.tokenize(marked_text)
            for i in range(0, len(tokenized_text), 510):
                tokenized_text = tokenized_text[i:i+block_size]
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                indexed_tokens = self.tokenizer.build_inputs_with_special_tokens(indexed_tokens)
                all_data.append(indexed_tokens)
                all_words.append(tokenized_text) 
                all_masks.append(list(np.ones(len(indexed_tokens))))
                all_users.append(sentence[0])

        lengths = np.array([len(l) for l in all_data])
        ordering = np.argsort(lengths)
        # each item in these lists is a sentence
        ordered_data = [None for i in range(len(all_data))]
        ordered_words = [None for i in range(len(all_data))]
        ordered_masks = [None for i in range(len(all_data))]
        ordered_users = [None for i in range(len(all_data))]
        for i, ind in enumerate(ordering): 
            ordered_data[i] = all_data[ind]
            ordered_words[i] = all_words[ind]
            ordered_masks[i] = all_masks[ind]
            ordered_users[i] = all_users[ind]
        # each item in these lists is a batch of sentences
        batched_data = []
        batched_words = []
        batched_masks = []
        batched_users = []
        i = 0
        current_batch = max_batch
        while i < len(ordered_data): 
            batch_data = ordered_data[i:i+current_batch]
            batch_words = ordered_words[i:i+current_batch]
            batch_mask = ordered_masks[i:i+current_batch]
            batch_users = ordered_users[i:i+current_batch]

            max_len = max([len(sent) for sent in batch_data])
            for j in range(len(batch_data)): 
                blen = len(batch_data[j])
                for k in range(blen, max_len): 
                    batch_data[j].append(0)
                    batch_mask[j].append(0)
            batched_data.append(torch.LongTensor(batch_data))
            batched_words.append(batch_words)
            batched_masks.append(torch.FloatTensor(batch_mask))
            batched_users.append(batch_users)
            i += current_batch
            if max_len > 100: 
                current_batch = 12
            if max_len > 200: 
                current_batch = 6
        return batched_data, batched_words, batched_masks, batched_users

    def get_embeddings(self, batched_data, batched_words, batched_masks, batched_users, word): 
        ret = [] 
        do_wordpiece = True
        print("Getting embeddings for batched_data of length", len(batched_data))
        for b in range(len(batched_data)):
            print("Batch #", b)
            # each item in these lists/tensors is a sentence
            tokens_tensor = batched_data[b].to(device)
            atten_tensor = batched_masks[b].to(device)
            words = batched_words[b]
            users = batched_users[b]
            with torch.no_grad():
                _, _, encoded_layers = self.model(tokens_tensor, attention_mask=atten_tensor, token_type_ids=None)
            print("batch size, sequence length, hidden layer size:", encoded_layers[0].size())
            for sent_i in range(len(words)): 
                for token_i in range(len(words[sent_i])):
                    if batched_masks[b][sent_i][token_i] == 0: continue
                    w = words[sent_i][token_i]
                    next_w = ''
                    if (token_i + 1) < len(words[sent_i]): 
                        next_w = words[sent_i][token_i+1]
                    if w != word and '##' not in w and '##' not in next_w: continue
                    if w == word: do_wordpiece = False
                    hidden_layers = [] 
                    for layer_i in range(1, 5):
                        vec = encoded_layers[-layer_i][sent_i][token_i]
                        hidden_layers.append(vec)
                    # concatenate last four layers
                    rep = torch.cat((hidden_layers[0], hidden_layers[1], 
                                hidden_layers[2], hidden_layers[3]), 0) 
                    ret.append((w, rep.cpu().numpy().reshape(1, -1)[0]))
        return (ret, do_wordpiece)

    def group_wordpiece(self, embeddings, word, do_wordpiece): 
        '''
        - puts together wordpiece vectors
        - only piece them together if embeddings does not 
        contain the vocab word of interest 
        - filters vectors so we only have vectors for the word of interest
        '''
        data = []
        for tup in embeddings: 
            if not do_wordpiece: 
                if tup[0] == word: 
                    data.append(tup[1])
            else: 
                # put together wordpiece word, gather vectors
                # if wordpiece word == word, average vectors ane append
        return np.array(data)

    def cluster_embeddings(self, data, word, dim_reduct=None, rs=0): 
        assert data.shape[0] >= 500, "Data isn't of size 500 but instead " + str(data.shape[0])
        if dim_reduct is not None:
            outpath = LOGS + 'reddit_pca/' + word + '_' + \
                str(dim_reduct) + '_' + str(rs) + '.joblib'
            pca = PCA(n_components=dim_reduct, random_state=rs)
            data = pca.fit_transform(data)
            dump(pca, outpath)
        ks = range(2, 10)
        centroids = {} 
        rss = np.zeros(len(ks))
        for i, k in enumerate(ks):
            km = KMeans(k, n_jobs=-1, random_state=rs)
            km.fit(data)
            rss[i] = km.inertia_
            centroids[k] = km.cluster_centers_
        crits = []
        lamb = 10000
        for i in range(len(ks)): 
            k = ks[i] 
            crit = rss[i] + lamb*k
            crits.append(crit)
        best_k = np.argmin(crits)
        return centroids[ks[best_k]] 

def main(): 
    word = sys.argv[1]
    doc = LOGS + 'vocabs/docs/' + word
    start = time.time()
    model = EmbeddingClusterer()
    instances = model.read_instances(doc)
    time.time()
    print("TOTAL TIME:", time1 - start)
    batched_data, batched_words, batched_masks, batched_users = model.get_batches(sentences, batch_size)
    time2 = time.time()
    print("TOTAL TIME:", time2 - time1)
    embeddings, do_wordpiece = model.get_embeddings(batched_data, batched_words, batched_masks, batched_users, word)
    time3 = time.time()
    print("TOTAL TIME:", time3 - time2)
    data = model.group_wordpiece(embeddings, word, do_wordpiece)
    centroids = model.cluster_embeddings(data, word, dim_reduct=100)
    np.save(LOGS + 'reddit_centroids/' + word + '.npy', centroids)


if __name__ == "__main__":
    main()