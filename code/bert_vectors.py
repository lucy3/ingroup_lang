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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False, do_basic_tokenize=False)
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
        with open(inputfile, 'r') as infile: 
            for line in infile: 
                contents = line.strip().split(' ')
                text = ' '.join(contents[1:])
                sentences.append((contents[0], "[CLS] " + text + " [SEP]"))
                if i > 100: break
                i += 1
        return sentences

    def get_batches(self, sentences, max_batch): 
        marked_text = sentences[0][1]
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)
        return tokens_tensor, tokenized_text

    def get_embeddings(self, tokens_tensor, tokenized_text): 
        with torch.no_grad():
            _, _, encoded_layers = self.model(tokens_tensor)
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
        # representation for a word is rep[word_index_in_sentence]
        print("Similarity between \'bank\' in \'bank vault\' and \'river bank\'")
        print(cosine_similarity(rep[6].cpu().reshape(1, -1), rep[19].cpu().reshape(1, -1)))
        print("Similarity between \'bank\' in \'bank robber\' and \'bank vault\'")
        print(cosine_similarity(rep[10].cpu().reshape(1, -1), rep[6].cpu().reshape(1, -1)))

if __name__ == "__main__":
    # TODO get input into correct format 
    filename = '/global/scratch/lucy3_li/ingroup_lang/subreddits_month/vegan/RC_2019-05'
    embeddings_model = BertEmbeddings()
    marked_text = embeddings_model.read_sentences(filename)
    tokens_tensor, tokenized_text = embeddings_model.get_batches(marked_text, batch_size)
    embeddings_model.get_embeddings(tokens_tensor, tokenized_text)
    '''
    # look at get_batches() in bamman-group-code
    # for each sentence get [tokens], [indexed tokens], [segment IDs], [attention mask], [batched transforms]
    # order the batches
    # get ordering 
    
    # run model on batches
    # for every token, get last four layers, concatenate
    # handle the wordpiece stuff somehow
    # filter out non-top-20% vocab
    # save as userID, word, representation
    '''
