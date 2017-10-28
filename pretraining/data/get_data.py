import spacy
import csv

nlp = spacy.load('en')

datapath = '/Users/siddharth/flipsideML/data/split/stories.final'
f = open(datapath, 'r').read()

articles = [tuple(line.split('\t')) for line in f.split('\n')]
del f

out_str = 'sentences\n'

article_count = 0
sentence_count = 0

for i in range(len(articles)):
    data = articles[i]
    if len(data) == 2:
        title, article = data[0], data[1]
        article_count += 1
        sents = nlp(article).sents
        for sent in sents:
            sentence_count += 1
            print('ARTICLES:{}'.format(article_count))
            print('SENTENCES:{}'.format(sentence_count))
            out_str += sent.text + '\n'
    else:
        pass

f = open('all_sentences.csv', 'w')

f.write(out_str)
