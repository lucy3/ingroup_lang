# for a subreddit
# get top 3% vocab, save to file (maybe do this in another file)
# run forward pass of BERT on all sentences containing top 3% vocab (keeping user info)
# get word vectors for every word in vocab and wordpiece
# put together wordpiece, only keep vocab words
# group together reps of the same words
# for each word and its reps, load centroid and match 
# output word_centroid# as sense for each user_line# into a subreddit-specific file
