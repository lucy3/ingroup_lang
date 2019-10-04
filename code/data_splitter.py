import os

large_files = ['askreddit', 'amitheasshole', 'politics']

def splitter(): 
    inpath = '/data0/lucy/ingroup_lang/subreddits/'
    outpath = '/data0/lucy/ingroup_lang/subreddits/'
    for sr in large_files: 
        print sr
        file_path = inpath + sr + '/RC_2019-05'
        with open(file_path, 'r') as file: 
            i = 1
            new_sr = sr + '@' + str(i)
            if not os.path.exists(outpath + new_sr): os.makedirs(outpath + new_sr)
            out_file = open(outpath + new_sr + '/RC_2019-05', 'w')
            j = 0
            for line in file: 
                if j > 1000000: 
                    j = 0
                    out_file.close()
                    i += 1
                    new_sr = sr + '@' + str(i)
                    if not os.path.exists(outpath + new_sr): os.makedirs(outpath + new_sr)
                    out_file = open(outpath + new_sr + '/RC_2019-05', 'w')
                out_file.write(line) 
                j += 1
                
def main(): 
    splitter()

if __name__ == '__main__':
    main()
