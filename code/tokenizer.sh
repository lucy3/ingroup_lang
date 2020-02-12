#!/bin/bash
# Tokenizes each subreddit's file

for filename in /data0/lucy/ingroup_lang/subreddits_month/*; do
    justfile=$(basename $filename)
    mkdir -p /data0/lucy/ingroup_lang/subreddits2/$justfile
    python tokenizer_helper.py $filename/RC_sample /data0/lucy/ingroup_lang/subreddits2/$justfile/
done
