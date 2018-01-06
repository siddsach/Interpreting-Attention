from io import open
import unicodedata
import string
import re
import random
import os
from nltk import sent_tokenize
from operator import itemgetter

nltk.download('punkt')

TRANSLATE = True

DATA_PATH = '/Users/siddharth/flipsideML/ML-research/deep/semi-supervised_clf/parse/data/'

UNNEEDED_LINES = 3



class PrepareData:

    def __init__(self, datapath = DATAPATH, language = 'english',WordVecPath = WORDVECPATH1, k = None, ):
        print('Reading Files and Extracting Sentences')
        sents, self.locs = self.get_sents(DATAPATH)
        print('Done extracting sentences')
       
        print('Building Vocabulary...')
        self.words = Lang('{}'.format(language)
        
        for sent in sents:
            sent = self.normalizeString(sent)
            self.words.addSentence(sent)
            del sent
        print('Done')

        print('Getting Word Vectors')
        self.words.set_glove_path(WordVecPath)
        self.words.getWordVecs(k)
        print('Retrieved all word vectors')


    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )



    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        
        #replace punctuation with consistent sentence end"
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!? ]+", r" ", s)
        return s


    def get_sents(self, root_path):
        sents = []
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
                        if sent:
                            locs[len(locs)] = {'folder':d, 'document':f, 'sentence', i}
                            sents.append(sent)
        return sents, locs

    def extract(self, f_path):
        f = open(f_path, 'r')
        for i in range(UNNEEDED_LINES):
            f.readline()
        text = f.read()
        text = clean(text)
        return [clean_sent(sent) for sent in sent_tokenize(text)]

    def clean(self, text):
        #ADD CODE TO CLEAN TEXT HERE IF NEEDED
        return text

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name, max_words):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.word2vec = {}
        self.index2word = {"UNK":0, "SOS":1,  "EOS":2}
        self.n_words = 3  # Count SOS and EOS
        self.sentlengths = []
        self.max_words = max_words

    def addSentence(self, sentence):
        words = sentence.split(' ')
        self.sentlengths.append(len(words))
        for word in words:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def getTopWords(self):
        sorted_words = sorted(self.word2count.items(), key = itemgetter(1))
        self.top_words = set()
        for word, _ in sorted_words:
            if len(top_words) <= len(self.max_words):
                self.top_words.add(word)

    def set_glove_path(self, path):
        self.glove_path = path

    def getWordVecs(self, K = None):
        assert hasattr(self, 'glove_path'), 'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        k = 0
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in self.top_words:
                    self.word2vec[word] = np.fromstring(vec, sep=' ')
        print('Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict)))
