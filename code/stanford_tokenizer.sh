#!/bin/bash
# Tokenizes each subreddit's file

for filename in /data0/lucy/ingroup_lang/subreddits_month/*; do
    justfile=$(basename $filename)
    mkdir -p /data0/lucy/ingroup_lang/subreddits2/$justfile
    java -cp "/data0/lucy/stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -file $filename/RC_sample -outputFormat conll -output.columns word -outputDirectory /data0/lucy/ingroup_lang/subreddits2/$justfile/
done
