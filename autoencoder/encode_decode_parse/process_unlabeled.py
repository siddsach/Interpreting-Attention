from io import open
import unicodedata
import string
import re
import random
import os
from nltk import sent_tokenize

nltk.download('punkt')

TRANSLATE = True

ROOT_PATH = '/Users/siddharth/flipsideML/ML-research/deep/semi-supervised_clf/parse/data/'
DATAPATH = ''
ALREADY_MADE_PAIRS = False

NEWPATH = ROOTPATH + 'data/pairs.txt'

LANG1 = 'english'
LANG2 = 'french'

UNNEEDED_LINES = 3

if TRANSLATE:
    DATAPATH = ROOTPATH + 'data/%s-%s.txt' % (lang1, lang2)
else:
    LANG2 = LANG1
    if not ALREADY_MADE_PAIRS:
        print("Don't have pair data for autoencoding. Building now...")
        data = get_pairs(DATAPATH, NEWPATH)
        print('Done.')
    DATAPATH = NEWPATH



def get_pairs(path, newpath):
    out_data = ''
    locs = {}
    folders = os.listdir(path)
    for d in folders:
        if d != '.DS_Store':
            d_path = path + '/' + d
            files = os.listdir(d_path)
            for f in files:
                f_path = d_path + '/' + f
                data = extract(f_path)
                for i, sent in enumerate(data):
                    locs[len(locs)] = {'folder':d, 'document':f, 'sentence', i}
                    out_data += sent + '\t' + sent + '\n'
    open(newpath, 'w').write(out_data)


def extract(f_path):
    f = open(f_path, 'r')
    for i in range(UNNEEDED_LINES):
        f.readline()
    text = f.read()
    text = clean(text)
    return sent_tokenize(text)

def clean(text):
    #ADD CODE TO CLEAN TEXT HERE IF NEEDED
    return text

'''
INPUT DATA FORMAT:

    TITLE: data/english-english.txt
    CONTENT: LINES WITH INPUT SENTENCE AND OUTPUT SENTENCE SEPARATED BY TAB

EXAMPLE:
    (TRANSLATION)
    I am cold.    Je suis froid.
    (AUTOENCODINGS)
    I am cold.    I am cold.
'''
class PrepareData:

    def __init__(self, translate = TRANSLATE, lang1 = LANG1, lang2 = LANG2, reverse=False, inputWordVecPath = WORDVECPATH1, outputWordVecPath = WORDVECPATH2, k = None):

        input_lang, output_lang, pairs = self.readLangs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))

        for pair in pairs:
            input_lang.addSentence(pair[0])
            if output_lang is not None:
                output_lang.addSentence(pair[1])

        print('Building Vocabulary...')
        input_lang.set_glove_path(inputWordVecPath)        
        input_lang.getWordVecs(k)

        if output_lang is not None:
            output_lang.set_glove_path(outputWordVecPath)            
            output_lang.getWordVecs(k)
        
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        if output_lang is not None:
            print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs



    def readLangs(self, lang1, lang2, reverse=False):
        print("Reading lines...")
        
        # Read the file and split into lines
        lines = open(DATAPATH, encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        input_lang = None
        output_lang = None
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            if lang1 == lang2:
                output_lang = None
            else:
                output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            if lang1 == lang2:
                output_lang = None
            else:
                output_lang = Lang(lang2)

        return input_lang, output_lang, pairs

    # Turn a Unicode string to plain ASCII, thanks to
    # http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters


    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s




SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.word2vec = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        words = []
        for word in sentence.split(' '):
            self.addWord(word)
            words.append(word)
        return words

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def set_glove_path(self, path):
        self.glove_path = path
    
    def getWordVecs(self, K = None):
        assert hasattr(self, 'glove_path'), 'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        k = 0
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in self.word2index:
                    if K is not None:
                        self.word2vec[word] = np.fromstring(vec, sep=' ')
                    else:
                        if k < K:
                            self.word2vec[word] = np.fromstring(vec, sep=' ')
                        else:
                            break
        print('Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict)))


