GIGAWORD_DATAPATHS = ['thread1.txt']

train_str = ''
val_str = ''
test_str = ''

split = 0.9
MAX_ARTICLES = 1000000

for path in GIGAWORD_DATAPATHS:


    print('READING {}...'.format(path))
    f = open(path, 'r')

    print('Loading data....')
    f = f.readlines()

    n_articles = len(f)

    for i, article in enumerate(f):

        print('READING ARTICLE {}'.format(i))
        data = article.split('\t')

        text = data[2]

        if i > min(MAX_ARTICLES, i):
            break

        if i < (split * n_articles):
            train_str += text
        elif i < ((split + ((1 - split) / 2)) * n_articles):
            val_str += text
        else:
            test_str += text

print('Writing to files...')
train_file = open('gigaword_train.txt', 'w')
val_file = open('gigaword_val.txt', 'w')
test_file = open('gigaword_test.txt', 'w')

train_file.write(train_str)
val_file.write(val_str)
test_file.write(test_str)
print('done')
