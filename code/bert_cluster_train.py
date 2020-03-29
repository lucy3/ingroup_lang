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
from functools import partial
from sklearn.decomposition import PCA
import random
import os.path
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

ROOT = '/global/scratch/lucy3_li/ingroup_lang/'
LOGS = ROOT + 'logs2/'

batch_size=32
dropout_rate=0.25
bert_dim=768


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingClusterer():
    def __init__(self, tokenizer, model_name):
        self.tokenizer = tokenizer
        self.model = model_name
        self.model.eval()
        self.model.to(device)

    def read_instances(self, inputfile): 
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
        # each item in these lists is a sentence
        all_data = [] # indexed tokens, or word IDs
        all_words = [] # tokenized_text, or original words
        all_masks = [] 
        all_users = []
        for sentence in sentences: 
            marked_text = sentence[1] 
            tokenized_text_all = self.tokenizer.tokenize(marked_text)
            for i in range(0, len(tokenized_text_all), 510):
                tokenized_text = tokenized_text_all[i:i+510]
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
        for b in range(len(batched_data)):
            # each item in these lists/tensors is a sentence
            tokens_tensor = batched_data[b].to(device)
            atten_tensor = batched_masks[b].to(device)
            words = batched_words[b]
            users = batched_users[b]
            with torch.no_grad():
                _, _, encoded_layers = self.model(tokens_tensor, attention_mask=atten_tensor, token_type_ids=None)
            for sent_i in range(len(words)): 
                for token_i in range(len(words[sent_i])):
                    if batched_masks[b][sent_i][token_i] == 0: continue
                    w = words[sent_i][token_i]
                    next_w = ''
                    if (token_i + 1) < len(words[sent_i]): 
                        next_w = words[sent_i][token_i+1]
                    if w != word and '##' not in w and '##' not in next_w: continue
                    if w == word: 
                        do_wordpiece = False
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
        prev_w = (None, None)
        ongoing_word = []
        ongoing_rep = []
        for i, tup in enumerate(embeddings):
            if not do_wordpiece: 
                if tup[0] == word: 
                    data.append(tup[1])
            else: 
                if tup[0].startswith('##'): 
                    if not prev_w[0].startswith('##'): 
                        ongoing_word.append(prev_w[0])
                        ongoing_rep.append(prev_w[1])
                    ongoing_word.append(tup[0][2:])
                    ongoing_rep.append(tup[1])
                else:
                    if ''.join(ongoing_word) == word: 
                        data.append(np.mean(ongoing_rep, axis=0).flatten())
                    ongoing_word = []
                    ongoing_rep = []
            prev_w = tup
        if ''.join(ongoing_word) == word: 
            data.append(np.mean(ongoing_rep, axis=0).flatten())
        np.random.shuffle(data)
        return np.array(data)[:500]

    def cluster_embeddings(self, data, ID, dim_reduct=None, rs=0, lamb=10000, finetuned=False): 
        assert data.shape[0] == 500, "Data isn't of size 500 but instead " + str(data.shape[0])
        if dim_reduct is not None:
            if finetuned: 
                outpath = LOGS + 'finetuned_reddit_pca/' + str(ID) + '_' + \
                   str(dim_reduct) + '_' + str(rs) + '.joblib'
            else: 
                outpath = LOGS + 'reddit_pca/' + str(ID) + '_' + \
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
        for i in range(len(ks)): 
            k = ks[i] 
            crit = rss[i] + lamb*k
            crits.append(crit)
        best_k = np.argmin(crits)
        return centroids[ks[best_k]] 

def main(): 
    word = sys.argv[1]
    print("WORD:", word)
    with open(LOGS + 'vocabs/vocab_map.json', 'r') as infile: 
        d = json.load(infile)
    ID = d[word]
    doc = LOGS + 'vocabs/docs/' + str(ID)
    finetuned = bool(int(sys.argv[2]))
    if finetuned: 
        print("Finetuned BERT-base")
        tokenizer = BertTokenizer.from_pretrained(LOGS + 'finetuning/', do_lower_case=True)
        model_name = BertModel.from_pretrained(LOGS + 'finetuning/', output_hidden_states=True) 
    else: 
        print("BERT-base")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model_name = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    start = time.time()
    model = EmbeddingClusterer(tokenizer, model_name)
    instances = model.read_instances(doc)
    time1 = time.time()
    batched_data, batched_words, batched_masks, batched_users = model.get_batches(instances, batch_size)
    time2 = time.time()
    embeddings, do_wordpiece = model.get_embeddings(batched_data, batched_words, batched_masks, batched_users, word)
    time3 = time.time()
    data = model.group_wordpiece(embeddings, word, do_wordpiece)
    time4 = time.time()
    centroids = model.cluster_embeddings(data, ID, dim_reduct=10, lamb=10000, finetuned=finetuned)
    if finetuned: 
        np.save(LOGS + 'finetuned_reddit_centroids/' + str(ID) + '.npy', centroids) 
    else: 
        np.save(LOGS + 'reddit_centroids/' + str(ID) + '.npy', centroids)
    time5 = time.time()
    print("TOTAL TIME:", time5 - start) 


if __name__ == "__main__":
    main()
