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
print(tokenized_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [1] * len(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
segments_tensors = torch.tensor([segments_ids]).to('cuda')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()
model.to('cuda')
with torch.no_grad():
    _, _, encoded_layers = model(tokens_tensor)
print(len(encoded_layers))
print("Batch size, tokens, hidden layer size:", encoded_layers[0].size())
token_embeddings = [] 
for token_i in range(len(tokenized_text)):
    hidden_layers = [] 
    for layer_i in range(1, 5):
        vec = encoded_layers[-layer_i][0][token_i]
        hidden_layers.append(vec)
    token_embeddings.append(hidden_layers)
rep = [torch.cat((layer[0], layer[1], layer[2], layer[3]), 0) for layer in token_embeddings]
# representation for a word is rep[word_index_in_sentence]
print(cosine_similarity(rep[6].cpu().reshape(1, -1), rep[19].cpu().reshape(1, -1)))
print(cosine_similarity(rep[10].cpu().reshape(1, -1), rep[6].cpu().reshape(1, -1)))

