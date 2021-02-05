Code for the TACL paper, _Characterizing English Variation across Social Media Communities with BERT_. Citation forthcoming. 

Lucy is currently adding comments and more information about this repo for easier use. 

The name of this repository if "ingroup_lang" since it is about in-group language. 

### Package versions
## Repository Map
### Code
This part is under construction, to be cleaned up for readibility :) 

Each file will be further commented at the top with details on inputs and purpose. The code will be further distilled down to the main methods and experiments presented in the final paper. It is currently a collection of everything Lucy has ever coded for this project. 

Data preprocessing, wrangling, and organizing
- data\_organize.py
- dataset\_statistics.py
- data\_splitter.py (deprecated over a year ago, to be deleted)
- get\_sense\_vocab.py
- langid.py
- language\_id.py
- language\_id\_helper.py
- mwe_discovery.py (not used, to be deleted)
- partial\_tokenizer.sh (deprecated over a year ago, to be deleted)
- remove\_duplicates.py
- rm\_dups\_single\_user.py (deprecated, to be deleted)
- run\_lm\_finetuning.py: (deprecated, to be deleted... we used Google's original BERT code instead) 
- slow\_tokenizer.py (possibly deprecated)

SemEval experiments
- bert\_vectors.py: get BERT embeddings
- bert\_post.py: piece together wordpiece vectors to create semeval2010_train_bert2 and semeval2010_test_bert2 files
- cluster\_vectors.py

k-means clustering and matching of word embeddings on Reddit data
- bert\_cluster\_train.py: clustering 1 word at a time
- bert\_cluster\_match.py: matching 1 subreddit at a time
- analyze\_bert.py: visualization
- playground.ipynb (purpose unclear, to be potentially deleted or merged with another notebook)

Amrami & Goldberg fork 
- link to public forked repo to be included here

Community language metrics
- sense\_pmi.py

Glossary analysis
- glossary\_eval.py
- senses.ipynb

Community behavior analysis
- comment\_networks.py
- comment\_networks\_helper.py
- loyalty.py

### Data
We used two months of data, May and June 2019, from (Pushshift's collection of Reddit comments)[https://files.pushshift.io/reddit/comments/]. 
If you would like the sampled comments (80k per subreddit) that Lucy used, email her since they are too big for Github. 
### Logs
This folder contains some of the outputs. 
