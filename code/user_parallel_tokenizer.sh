#!/bin/bash

# Usage: ./parallel_tokenizer.sh &> ../logs/token.temp

for filename in /data0/lucy/ingroup_lang/subreddits/*; do
    justfile=$(basename $filename)
    mkdir -p /data0/lucy/ingroup_lang/subreddits2/$justfile
    file_list=/data0/lucy/ingroup_lang/logs/file_lists/file_list_$justfile.txt
    for user_month in /data0/lucy/ingroup_lang/subreddits/filename/*; do 
        find $filename/ -mindepth 1 -maxdepth 1 -name "*" > $file_list
    done 
    java -cp "/data0/lucy/stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -threads 10 -fileList $file_list -outputFormat conll -output.columns word -outputDirectory /data0/lucy/ingroup_lang/subreddits2/$justfile/
done
