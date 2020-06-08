#!/bin/bash
# Tokenizes each subreddit's file
# After first round of reviews, added POS tagging as well 

for filename in /mnt/data0/lucy/ingroup_lang/subreddits_month/*; do
    justfile=$(basename $filename)
    mkdir -p /mnt/data0/lucy/ingroup_lang/subreddits_pos/$justfile
    java -cp "/mnt/data0/lucy/stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos -file $filename/RC_sample -outputFormat conll -output.columns word,pos -ssplit.newlineIsSentenceBreak always -outputDirectory /mnt/data0/lucy/ingroup_lang/subreddits_pos/$justfile/
done
