"""
TODO: remove duplicate words for each user in
tokenized file. This means if we see a line that starts
with @@#_USER#@@_, we create a new set. We ignore username
lines as well when outputting to a new file. 

This file should be parallelized across subreddits if possible
and written similarly to rm_dups_single_user.py 
"""
