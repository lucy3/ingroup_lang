import os,sys,argparse
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
        sentences = []
        i = 0
        curr_user = None
        with open(inputfile, 'r') as infile: 
            for line in infile: 
                contents = line.strip()
                if contents.startswith('@@#USER#@@_'): 
                    curr_user = contents
                else: 
                    sentences.append((curr_user, "[CLS] " + contents + " [SEP]"))
                    if i > 10: break
                    i += 1
        return sentences

    def get_batches(self, sentences, max_batch): 
        all_data = [] # indexed tokens, or word IDs
        all_words = [] # tokenized_text, or original words
        all_masks = [] 
        all_users = []
        for sentence in sentences: 
            marked_text = sentences[1]
            print(marked_text)
            tokenized_text = self.tokenizer.tokenize(marked_text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            all_data.append(indexed_tokens)
            all_words.append(tokenized_text) 
            all_masks.append(np.ones(len(indexed_tokens)))
            all_users.append(sentences[0])

        lengths = np.array([len(l) for l in all_data])
        ordering = np.argsort(lengths)
        ordered_data = [None for i in range(len(all_data))]
        ordered_words = [None for i in range(len(all_data))]
        ordered_masks = [None for i in range(len(all_data))]
        ordered_users = [None for i in range(len(all_data))]
        
        return 

    def get_embeddings(self, indexed_tokens, tokenized_text): 
        tokens_tensor = indexed_tokens.to(device)
        # TODO to(device) for other inputs into self.model as well
        with torch.no_grad():
            _, _, encoded_layers = self.model(tokens_tensor, attention_mask=TODO, token_type_ids=None)
        print("Batch size, tokens, hidden layer size:", encoded_layers[0].size())
        token_embeddings = [] 
        for token_i in range(len(tokenized_text)):
            hidden_layers = [] 
            for layer_i in range(1, 5):
                vec = encoded_layers[-layer_i][0][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        # concatenate last four layers
        rep = [torch.cat((layer[0], layer[1], layer[2], layer[3]), 0) for layer in token_embeddings]
        for i in range(len(rep)):
            # TODO: check if in subreddit's vocabulary, or maybe just save everything?
            # TODO: keep track of user as well
            print(tokenized_text[i], rep[i].cpu().reshape(1, -1))

if __name__ == "__main__":
    filename = '/global/scratch/lucy3_li/ingroup_lang/subreddits_month/vegan/RC_2019-05'
    embeddings_model = BertEmbeddings()
    marked_text = embeddings_model.read_sentences(filename)
    indexed_tokens, tokenized_text = embeddings_model.get_batches(marked_text, batch_size)
    embeddings_model.get_embeddings(indexed_tokens, tokenized_text)
