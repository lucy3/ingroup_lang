'''
Getting vectors for SemEval train and test sets 
I did not end up using this file to run on Reddit 
data because the way it was set up
makes it so that it is too slow on larger datasets
'''
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
from nltk.stem import WordNetLemmatizer

root = '/global/scratch/lucy3_li/'
#root = '/data0/lucy/'
sem_eval_trial_data = '../semeval-2012-task-13-trial-data/data/semeval-2013-task-10-trial-data.xml'
sem_eval_train = root + 'ingroup_lang/logs/ukwac2.txt' 
sem_eval_test = '../SemEval-2013-Task-13-test-data/contexts/xml-format/'
sem_eval_2010_train = root + 'ingroup_lang/semeval-2010-task-14/training_data/'
sem_eval_2010_test = root + 'ingroup_lang/semeval-2010-task-14/test_data/'

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
 
    def read_semeval2010_test_sentences(self): 
        print("Reading sem eval 2010 test sentences...")
        sentences = []
        for pos in ['nouns', 'verbs']: 
            for f in os.listdir(sem_eval_2010_test + pos):
                tree = ET.parse(sem_eval_2010_test + pos + '/' + f)
                root = tree.getroot()
                lemma = f.replace('.xml', '') 
                for instance in root: 
                    tag = instance.tag
                    ID = tag + '_' + lemma + '_' + tag.split('.')[0]
                    for child in instance: 
                        assert child.tag == 'TargetSentence'
                        sent = child.text
                        sentences.append((ID, "[CLS] " + sent + " [SEP]"))
        return sentences

    def read_semeval2010_train_sentences(self):
        """
        ID is id_lemma.pos_lemma
        """
        print("Reading sem eval 2010 train sentences...")
        sentences = []
        for pos in ['verbs', 'nouns']: 
            for f in os.listdir(sem_eval_2010_train + pos): 
                tree = ET.parse(sem_eval_2010_train + pos + '/' + f)
                root = tree.getroot()
                lemma = f.replace('.xml', '')
                for instance in root: 
                    tag = instance.tag
                    ID = tag + '_' + lemma + '_' + tag.split('.')[0]
                    sent_tok = sent_tokenize(instance.text)
                    for sent in sent_tok: 
                        sentences.append((ID, "[CLS] " + sent + " [SEP]"))
        return sentences

    def read_semeval_test_sentences(self): 
        """
        Each word has its own xml file. 
        ID is id_lemma.pos_token
        """
        print("Reading sem eval 2013 test sentences...") 
        sentences = []
        for f in os.listdir(sem_eval_test): 
            tree = ET.parse(sem_eval_test + f)
            root = tree.getroot()
            lemma = f.replace('.xml', '')
            for instance in root: 
                ID = instance.attrib['id'] + '_' + lemma + '_' + instance.attrib['token'].lower() 
                sent = instance.text
                sentences.append((ID, "[CLS] " + sent + " [SEP]"))
        return sentences

    def read_semeval_train_sentences(self):
        '''
        The ID is the linenumber_lemma.pos_token 
        '''
        print("Reading sem eval 2013 train sentences...")
        sentences = []
        with open(sem_eval_train, 'r') as infile: 
            i = 0
            for line in infile: 
                contents = line.strip().split('\t') 
                ID = str(i) + '_' + contents[0] + '_' + contents[1]
                sent = ' '.join(contents[2:])
                i += 1
                sentences.append((ID, "[CLS] " + sent + " [SEP]"))
        return sentences
    
    def read_semeval_trial_sentences(self): 
        '''
        This input file is a little wonky; see sem eval readme for more details. 
        The output of sentences contains a list of tuples
        where the first tuple is the instance id and the second is the sentence.  
        '''
        print("Reading sem eval sentences...") 
        sentences = []
        tree = ET.parse(sem_eval_trial_data)
        root = tree.getroot()
        for instance in root: 
            ID = instance.attrib['id'] + '_' + instance.attrib['lemma'] + '_' + instance.attrib['token'].lower() 
            sent = instance.text
            sentences.append((ID, "[CLS] " + sent + " [SEP]"))
        return sentences

    def read_sentences(self, inputfile): 
        '''
        The input file is formatted 
        where each line is a single input sentence
        or a username. Usernames come before their sentences.  
        Each input file is a single subreddit. 
        '''
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
                        sentences.append((str(line_number) + '_' + curr_user, "[CLS] " + sent + " [SEP]"))
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

    def get_embeddings(self, batched_data, batched_words, batched_masks, \
              batched_users, outfile, only_save_lemmas=False, comb_layers=False, last_layer=False): 
        wnl = WordNetLemmatizer()
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
                    w = words[sent_i][token_i]
                    if only_save_lemmas: 
                        # only save lemmas and wordpieces, to save space
                        next_w = ''
                        if (token_i + 1) < len(words[sent_i]): 
                            next_w = words[sent_i][token_i+1]
                        if '##' not in w and '##' not in next_w: 
                            ID = users[sent_i].split('_')
                            lemma = ID[-2].split('.')[0]
                            pos = ID[-2].split('.')[1]
                            if pos == 'j': pos = 'a'
                            if wnl.lemmatize(w, pos) != lemma: continue
                    if w == '[CLS]' or w == '[SEP]': continue
                    if comb_layers: 
                        hidden_layers = []
                        for layer_i in range(1, 13): 
                            vec = encoded_layers[-layer_i][sent_i][token_i]
                            hidden_layers.append(vec)
                        rep = torch.sum(torch.stack(hidden_layers), dim=0) # change to mean for avg
                    elif last_layer: 
                        rep = encoded_layers[-1][sent_i][token_i]
                    else: 
                        hidden_layers = [] 
                        for layer_i in range(1, 5):
                            vec = encoded_layers[-layer_i][sent_i][token_i]
                            hidden_layers.append(vec)
                        # concatenate last four layers
                        rep = torch.cat((hidden_layers[0], hidden_layers[1], 
                                hidden_layers[2], hidden_layers[3]), 0) 
                    ofile.write(users[sent_i] + '\t' +  w + '\t' + \
                            ' '.join(str(n) for n in rep.cpu().numpy().reshape(1, -1)[0]) + '\n')
        ofile.close()

def run_bert_on_reddit(): 
    root_path = root + 'ingroup_lang/'
    for subreddit in ['askreddit']: 
        filename = root_path + 'subreddits_month/' + subreddit + '/RC_sample'
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

def run_bert_on_semeval(test=False, twentyten=False, only_save_lemmas=False, comb_layers=False, last_layer=False):
    root_path = root + 'ingroup_lang/' 
    start = time.time()
    embeddings_model = BertEmbeddings()
    if test: 
        if twentyten:
            outfile = root_path + 'logs/semeval2010/semeval2010_test_bert_last'
            sentences = embeddings_model.read_semeval2010_test_sentences()
        else: 
            outfile = root_path + 'logs/semeval2013/semeval2013_test_bert'
            sentences = embeddings_model.read_semeval_test_sentences()
    else: 
        if twentyten: 
            outfile = root_path + 'logs/semeval2010/semeval2010_train_bert_last' 
            sentences = embeddings_model.read_semeval2010_train_sentences()
        else: 
            outfile = root_path + 'logs/semeval2013/semeval2013_train_bert'
            sentences = embeddings_model.read_semeval_train_sentences()
    time1 = time.time()
    print("TOTAL TIME:", time1 - start)
    batched_data, batched_words, batched_masks, batched_users = embeddings_model.get_batches(sentences, batch_size)
    time2 = time.time()
    print("TOTAL TIME:", time2 - time1)
    embeddings_model.get_embeddings(batched_data, batched_words, batched_masks, batched_users, \
            outfile, only_save_lemmas=only_save_lemmas, comb_layers=comb_layers, last_layer=last_layer)
    print("TOTAL TIME:", time.time() - time2)

 
def main(): 
    run_bert_on_semeval(test=False, twentyten=True, only_save_lemmas=True)
    run_bert_on_semeval(test=True, twentyten=True, only_save_lemmas=True)

if __name__ == "__main__":
    main()
