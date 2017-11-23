import tinys3

keypath = '/Users/siddharth/flipsideML/data/s3key.csv'
splits = ['train', 'val', 'test']
gigaword_paths = ['gigaword_{}.txt'.format(s) for s in splits]

for path in gigaword_paths:

    print('Connecting to s3.')
    keys = open(keypath, 'r').read().split('\n')

    access = keys[0][15:]
    secret = keys[1][13:]
    conn = tinys3.Connection(access, secret)
    print('Connected with S3.')

    print('uploading...')
    f = open(path, 'rb')
    conn.upload(path, f, 'gigaword')
    print('Done')
