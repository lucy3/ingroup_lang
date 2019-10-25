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

batch_size=32
dropout_rate=0.25
bert_dim=768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('')
print("********************************************")
print("Running on: {}".format(device))
print("********************************************")
print('')

"""
Built off of the following tutorial: 
https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
"""

class BertEmbeddings():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.model.to(device)

    def read_sentences(self, inputfile): 
        '''
        The input file is formatted as 
        userID sentence
        where each line is a single input sentence. 
        Each input file is a single subreddit. 
        '''
        print("Reading sentences...")
        sentences = []
        i = 0
        curr_user = None
        with open(inputfile, 'r') as infile: 
            for line in infile: 
                contents = line.strip()
                if contents.startswith('@@#USER#@@_'): 
                    curr_user = contents
                else:
                    sent_tok = sent_tokenize(contents)
                    for sent in sent_tok: 
                        sentences.append((curr_user, "[CLS] " + sent + " [SEP]"))
                        i += 1
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
            tokenized_text = tokenized_text[:512]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
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

    def get_embeddings(self, batched_data, batched_words, batched_masks, batched_users, outfile): 
        ofile = open(outfile, 'w')
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
                    # TODO if we filter by vocabulary, do it here
                    hidden_layers = [] 
                    for layer_i in range(1, 5):
                        vec = encoded_layers[-layer_i][sent_i][token_i]
                        hidden_layers.append(vec)
                    # concatenate last four layers
                    rep = torch.cat((hidden_layers[0], hidden_layers[1], 
                                hidden_layers[2], hidden_layers[3]), 0) 
                    if words[sent_i][token_i] == '[CLS]' or words[sent_i][token_i] == '[SEP]': continue
                    ofile.write(users[sent_i] + '\t' +  words[sent_i][token_i] + '\t' + \
                            ' '.join(str(n) for n in rep.cpu().numpy().reshape(1, -1)[0]) + '\n')
        ofile.close()

if __name__ == "__main__":
    root_path = '/global/scratch/lucy3_li/ingroup_lang/'
    subreddit = 'vegan'
    filename = root_path + 'subreddits_month/' + subreddit + '/RC_2019-05'
    start = time.time()
    embeddings_model = BertEmbeddings()
    sentences = embeddings_model.read_sentences(filename)
    time1 = time.time()
    print("TOTAL TIME:", time1 - start)
    batched_data, batched_words, batched_masks, batched_users = embeddings_model.get_batches(sentences, batch_size)
    time2 = time.time()
    print("TOTAL TIME:", time2 - time1)
    outfile = root_path + 'logs/bert_vectors/' + subreddit
    embeddings_model.get_embeddings(batched_data, batched_words, batched_masks, batched_users, outfile)
    print("TOTAL TIME:", time.time() - time2)
