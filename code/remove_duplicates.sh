#!/bin/bash

for filename in /data0/lucy/ingroup_lang/subreddits2/*; do
    justfile=$(basename $filename)
    output_folder=/data0/lucy/ingroup_lang/subreddits3/$justfile
    mkdir -p $output_folder
    file_list=/data0/lucy/ingroup_lang/logs/file_lists/file_list_$justfile.txt
    for user_month in /data0/lucy/ingroup_lang/subreddits2/filename/*; do 
        find $filename/ -mindepth 1 -maxdepth 1 -name "*" > $file_list
    done 
    cat $file_list | parallel --jobs 10 'python remove_duplicates.py {}' &
    wait
    output_folder2=/data0/lucy/ingroup_lang/subreddits4/$justfile
    mkdir -p $output_folder2
    for f in $output_folder/RC_2019-05_*.conll; do cat "$f" >> $output_folder2/RC_2019-05; done
done