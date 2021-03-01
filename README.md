# Reddit Language Variation

Code for the TACL paper, _Characterizing English Variation across Social Media Communities with BERT_. Citation forthcoming. 

Lucy is currently adding comments and more information about this repo for easier use. 

The name of this repository if "ingroup_lang" since it is about in-group language. 

### Package versions

Apache Spark 2.4.3, PyTorch 1.6.0, [transformers](https://huggingface.co/transformers/) 3.3.1, Python 3.7 

See requirements.txt for more details. Some code from early on in the project may be in Python 2.7, but I have tried to upgrade all instances to Python 3.7, but may have missed some, just let me know. 

## Repository Map
### Code

**Data preprocessing, wrangling, and organizing**
- data\_organize.py
- dataset\_statistics.py
- get\_sense\_vocab.py
- langid.py
- language\_id.py
- language\_id\_helper.py
- tokenizer.sh
- tokenizer\_helper.py

**SemEval experiments**
- bert\_vectors.py
- bert\_post.py
- cluster\_vectors.py

**Clustering and matching of word embeddings on Reddit data**
- bert\_cluster\_train.py: clustering 1 word at a time
- bert\_cluster\_match.py: matching 1 subreddit at a time
- analyze\_bert.py: visualization
- spectral.py

**Amrami & Goldberg 2019 fork**
- [The repo here](https://github.com/lucy3/bertwsi)
- Thank you to Asaf Amrami for making your code accessible 

**Community language metrics**
- sense\_pmi.py
- textrank.py
- word\_rarity.py

**Glossary analysis**
- glossary\_eval.py
- senses.ipynb

**Community behavior analysis**
- comment\_networks.py
- comment\_networks\_helper.py
- loyalty.py
- sociolect\_score\_analysis.ipynb
- users.py
- users\_sociolect\_analysis.py

### Data
We used two months of data, May and June 2019, from [Pushshift's collection of Reddit comments](https://files.pushshift.io/reddit/comments/). 
If you would like the sampled comments (80k per subreddit) that Lucy used, email her since they are too big for Github. 

Download SemEval 2013 Task 13 data: [here](https://www.cs.york.ac.uk/semeval-2013/task13/). You should get a folder called "SemEval-2013-Task-13-test-data" that contains test data. 
The ukwac corpus for training SemEval 2013 can be found [here](https://wacky.sslmit.unibo.it/doku.php?id=download), you may need to contact the owners to get a downloaded version. 
Download SemEval 2010 Task 14 data: [here](https://www.cs.york.ac.uk/semeval2010_WSI/datasets.html). You should get a folder called "semeval-2010-task-14" that contains training and test data. 

Subreddit glossaries, as csvs, are also in this folder. 

### Logs
This folder contains some of the outputs. There are several files also listing some of the community attributes of each subreddit in our dataset. 

- base\_most\_sense\_pmi are pmi scores, largest to smallest, for BERT-base k-means
- ag\_most\_sense\_pmi are pmi scores, largest to smallest, for Amrami & Goldberg model 
- norm\_pmi are type pmi scores, smallest to largest 
