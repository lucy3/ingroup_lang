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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
marked_text = "[CLS] " + text + " [SEP]"
# SINGLE SENTENCE EXAMPLE
tokenized_text = tokenizer.tokenize(marked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [1] * len(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
token_embeddings = [] 
for token_i in range(len(tokenized_text)):
    hidden_layers = [] 
    for layer_i in range(1, 5):
        # TODO MODIFY THIS SO WE JUST NEED THE LAST FOUR LAYERS
        vec = encoded_layers[-layer_i][batch_i][token_i]
        hidden_layers.append(vec)
    token_embeddings.append(hidden_layers)
rep = [torch.cat((layer[0], layer[1], layer[2], layer[3]), 0) for layer in token_embeddings]
# representation for a word is rep[word_index_in_sentence]
print(cosine_similarity(rep[6].reshape(1, -1), rep[19].reshape(1, -1)))
print(cosine_similarity(rep[10].reshape(1, -1), rep[6].reshape(1, -1)))
