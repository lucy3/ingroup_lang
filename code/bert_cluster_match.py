# for a subreddit
# run forward pass of BERT on all sentences containing top 3% vocab (keeping user info)
# get word vectors for every word in vocab and wordpiece
# put together wordpiece, only keep vocab words
# group together reps of the same words
# for each word and its reps, load centroid and match 
# output word_centroid# as sense for each user_line# into a subreddit-specific file

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

class EmbeddingMatcher():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.model.to(device)

    def read_sentences(self, inputfile): 
        print("Reading sentences...")
        sentences = []
        i = 0
        curr_user = None
        line_number = 0
        with open(inputfile, 'r') as infile: 
            for line in infile: 
                contents = line.strip()
                if contents.startswith('USER1USER0USER'): 
                    curr_user = contents
                else:
                    sent_tok = sent_tokenize(contents)
                    for sent in sent_tok: 
                        sentences.append((str(line_number) + '_' + curr_user, sent))
                        i += 1
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
                tokenized_text = tokenized_text[i:i+510]
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
    
    def get_embeddings(self, batched_data, batched_words, batched_masks, batched_users, vocab): 
        ret = [] 
        do_wordpiece = True
        print("Getting embeddings for batched_data of length", len(batched_data))
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
                    if w not in vocab and '##' not in w and '##' not in next_w: continue
                    hidden_layers = [] 
                    for layer_i in range(1, 5):
                        vec = encoded_layers[-layer_i][sent_i][token_i]
                        hidden_layers.append(vec)
                    # concatenate last four layers
                    rep = torch.cat((hidden_layers[0], hidden_layers[1], 
                                hidden_layers[2], hidden_layers[3]), 0) 
                    ret.append((w, users[sent_i], rep.cpu().numpy().reshape(1, -1)[0]))
        return (ret, do_wordpiece)

    def group_wordpiece(self, embeddings, vocab): 
        '''
        - puts together wordpiece vectors
        - filters vectors so we only have vectors for the word of interest
        - groups together representations of the same word
        '''
        print("Grouping wordpiece vectors: " + str(do_wordpiece))
        data = defaultdict(list) # { word : [(user, rep)] } 
        prev_w = (None, None, None)
        ongoing_word = []
        ongoing_rep = []
        for i, tup in enumerate(embeddings):
            if not tup[0].startswith('##'):    
                if len(ongoing_word) == 0 and prev_w[0] is not None:
                    data[prev_w[0]].append((prev_w[1], prev_w[2]))
                elif prev_w[0] is not None:          
                    rep = np.array(ongoing_rep)
                    rep = np.mean(rep, axis=0).flatten()
                    tok = ''
                    for t in ongoing_word: 
                        if t.startswith('##'): t = t[2:]
                        tok += t
                    if tok in vocab: 
                        data[tok].append((prev_w[1], rep))
                ongoing_word = []
                ongoing_rep = []
            else: 
                if len(ongoing_word) == 0 and prev_line is not None: 
                    ongoing_word.append(prev_w[0])
                    ongoing_word_rep.append(prev_w[2])
                ongoing_word.append(tup[0])
                ongoing_word_rep.append(tup[2])
            prev_w = tup
        if len(ongoing_word) == 0 and prev_w[0] is not None: 
            data[prev_w[0]].append((prev_w[1], prev_w[2]))
        elif prev_w[0] is not None:
            rep = np.array(ongoing_word_rep)
            rep = np.mean(rep, axis=0).flatten()
            tok = ''
            for t in ongoing_word: 
                if t.startswith('##'): t = t[2:]
                tok += t
            if tok in vocab: 
                data[tok].append((prev_w[1], rep)) 
        return data

    def match_embeddings(self, data, word, dim_reduct=None, rs=0): 
    '''
    for each word and its reps, load centroid and match 
    output: line#\tuser\tword\tcentroid#\n in a subreddit-specific file
    '''
    # open output file
    for token in data: 
        # load centroids 
        reps = 
    # close output file  

def main(): 
    subreddit = sys.argv[1]
    vocab = # TODO get vocab from subreddit vocab file
    fake_vocab = set(['dinosaurs', 'fit', 'insecurity'])  
    vocab = set(vocab) & set(fake_vocab) 
    model = EmbeddingMatcher()
    

if __name__ == "__main__":
    main()
