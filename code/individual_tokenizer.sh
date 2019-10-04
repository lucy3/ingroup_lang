#!/bin/bash

# Tokenizes individual subreddit parts
# For example, if run as ./individual_tokenizer askreddit, would tokenize
# the folders askreddit@1, askreddit@2, etc. 

for filename in /data0/lucy/ingroup_lang/subreddits/*; do 
	justfile=$(basename $filename)
        if [[ $justfile == "$1@"* ]] 
        then
		mkdir -p /data0/lucy/ingroup_lang/subreddits2/$justfile
		java -cp "/data0/lucy/stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -file $filename/RC_2019-05 -outputFormat conll -output.columns word -outputDirectory /data0/lucy/ingroup_lang/subreddits2/$justfile/
	fi
done 
