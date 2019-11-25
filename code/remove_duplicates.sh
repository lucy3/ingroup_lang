#!/bin/bash

# create filelist of files to run remove_duplicates.py on 
# run parallel on these files
file_list=/data0/lucy/ingroup_lang/logs/file_list.txt
echo -n > $file_list
for filename in /data0/lucy/ingroup_lang/subreddits2/*; do
    justfile=$(basename $filename)
    output_folder=/data0/lucy/ingroup_lang/subreddits3/$justfile
    mkdir -p $output_folder
    echo "/data0/lucy/ingroup_lang/subreddits2/$justfile/RC_sample.conll" >> $file_list
done
cat $file_list | parallel --jobs 10 'python remove_duplicates.py {}' &
wait
