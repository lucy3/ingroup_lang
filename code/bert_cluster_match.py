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

class EmbeddingMatcher():
    def __init__(self, tokenizer, model_name):
        self.tokenizer = tokenizer
        self.model = model_name
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
        return ret

    def group_wordpiece(self, embeddings, vocab): 
        '''
        - puts together wordpiece vectors
        - filters vectors so we only have vectors for the word of interest
        - groups together representations of the same word
        '''
        print("Grouping wordpiece vectors...")
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
                if len(ongoing_word) == 0 and prev_w[0] is not None: 
                    ongoing_word.append(prev_w[0])
                    ongoing_rep.append(prev_w[2])
                ongoing_word.append(tup[0])
                ongoing_rep.append(tup[2])
            prev_w = tup
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
        return data

    def match_embeddings(self, data, vocab, subreddit, viz=False, dim_reduct=10, rs=0, finetuned=False): 
        '''
        for each word and its reps, load centroid and match 
        output: line#_user\tword\tcentroid#\n in a subreddit-specific file
        '''
        if finetuned: 
            outfile = open(LOGS + 'finetuned_senses/' + subreddit, 'w')
            centroids_folder = LOGS + 'finetuned_reddit_centroids/' 
            pca_folder = LOGS + 'finetuned_reddit_pca/' 
        else: 
            outfile = open(LOGS + 'senses/' + subreddit, 'w')
            centroids_folder = LOGS + 'reddit_centroids/' 
            pca_folder = LOGS + 'reddit_pca/'
        for token in data: 
            assert token in vocab, "This token " + token + " is not in the vocab!!!!"
            centroids = np.load(centroids_folder + token + '.npy') 
            rep_list = data[token]
            IDs = []
            reps = []
            for tup in rep_list: 
                IDs.append(tup[0])
                reps.append(tup[1])
            reps = np.array(reps)
            inpath = pca_folder + token + \
             '_' + str(dim_reduct) + '_' + str(rs) + '.joblib'
            pca = load(inpath)
            reps = pca.transform(reps)
            assert reps.shape[1] == centroids.shape[1] 
            sims = cosine_similarity(reps, centroids) # IDs x n_centroids
            labels = np.argmax(sims, axis=1)
            for i in range(len(IDs)): 
                if viz:
                    outfile.write(IDs[i] + '\t' + token + '\t' + str(labels[i]) + '\t' + \
                       ' '.join(str(n) for n in reps[i]) + '\n')  
                else: 
                    outfile.write(IDs[i] + '\t' + token + '\t' + str(labels[i]) + '\n') 
        outfile.close()

def main(): 
    subreddit = sys.argv[1]
    print(subreddit)
    inputfile = ROOT + 'subreddits_month/' + subreddit + '/RC_sample'
    vocab = set()
    # TODO: change to only matching words in subreddit's sense vocab & 10_1_filtered
    with open(LOGS + 'vocabs/tiny_vocab', 'r') as infile: 
        for line in infile: 
            vocab.add(line.strip())
    start = time.time()
    finetuned = bool(sys.argv[2])
    if finetuned: 
        print("Finetuned BERT")
        tokenizer = BertTokenizer.from_pretrained(LOGS + 'finetuning/', do_lower_case=True)
        model_name = BertModel.from_pretrained(LOGS + 'finetuning/', output_hidden_states=True) 
    else: 
        print("BERT-base")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model_name = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model = EmbeddingMatcher(tokenizer, model_name)
    sentences = model.read_sentences(inputfile) 
    time1 = time.time()
    print("TOTAL TIME:", time1 - start)
    batched_data, batched_words, batched_masks, batched_users = model.get_batches(sentences, batch_size)
    time2 = time.time()
    print("TOTAL TIME:", time2 - time1)
    embeddings = model.get_embeddings(batched_data, batched_words, batched_masks, batched_users, vocab)
    time3 = time.time()
    print("TOTAL TIME:", time3 - time2)
    data = model.group_wordpiece(embeddings, vocab)
    time4 = time.time()
    print("TOTAL TIME:", time4 - time3)
    model.match_embeddings(data, vocab, subreddit, dim_reduct=10, rs=0, finetuned=finetuned)
    time5 = time.time()
    print("TOTAL TIME:", time5 - time4) 
    

if __name__ == "__main__":
    main()