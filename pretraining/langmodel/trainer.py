from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe, CharNGram
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from model import LangModel
import time
#GET THAT FINE-ASS NOISE CONTRASTIVE SAMPLING MMMM YOU TRAIN MY LANGUAGE MODELS SO GOOD
from nce import NCELoss

RAW_TEXTDATA_PATH = '/Users/siddharth/flipsideML/ML-research/deep/semi-supervised_clf/pretraining/data/more_sentences.csv' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
VECTOR_FOLDER = '/Users/siddharth/flipsideML/ML-research/deep/semi-supervised_clf/vectors'
VECTOR_CACHE = 'vectors'
SAVED_VECTORS = False
NUM_EPOCHS = 10
BPTT_LENGTH = 35
LEARNING_RATE = 0.5
BATCH_SIZE = 50
LOG_INTERVAL = 5
BPTT_SEQUENCE_LENGTH = 2
WORDVEC_DIM = 300
WORDVEC_SOURCE = ['GloVe']# charLevel']
DEFAULT_DATAPATH = 'wikitext-2/wiki.train.tokens'
MODEL_SAVE_PATH = 'langmodel.pt'

'''
CREATING A POSTPROCESSING FUNCTION TO TURN SEQUENCES OF
INDEXES FOR EACH WORD INTO WORD VECTORS
'''
def preprocess(x):

    try:
        return x.encode('utf-8').decode('utf-8').lower()
    except ValueError:
        print(type(x))
        print(x)
    #out = torch.zeros(len(x), WORD_VEC_DIM)
    #if train:
    #    c = 0
    #    for i, word in enumerate(x):
    #        out[i] = vocab.vectors[word]
    #        c+= 1
    #        if c > 128:
    #            break
    #    print('FINISHED TURNING INTO VECTORS')
    #    print(type(out))
    #    print(out.shape)
    #    pickle.dump(out, open('weird_tensor.p', 'wb'))
    #    out = [torch.FloatTensor(out) for el in out]
    #    print('FINISHED CONVERTING')
    #    return out
    # '''
    #out = arr
    #for i, example in enumerate(arr):
    #    new_example = len(example) * [None]
    #    for j, word in enumerate(example):
    #        vec = vocab.vectors[word]
    #        new_example[j] = vec
    #    out[i] = new_example
    #print('Done preprocessing')
    #return out
    #'''

def get_freqs(vocab_object):

    num_specials = 4
    vocab_size = len(vocab_object.itos) - num_specials
    out = torch.zeros(vocab_size)

    for i in range(num_specials, vocab_size + num_specials):
        out[i] = vocab_object.freqs[vocab_object.itos[i]]

    return out


def build_unigram_noise(freq):
    """build the unigram noise from a list of frequency
    Parameters:
        freq: a tensor of #occurrences of the corresponding index
    Return:
        unigram_noise: a torch.Tensor with size ntokens,
        elements indicate the probability distribution
    """
    total = freq.sum()
    noise = freq / total
    assert abs(noise.sum() - 1) < 0.001
    return noise


class TrainLangModel:
    def __init__(
                    self,
                    datapath = None, #RAW_TEXTDATA_PATH,
                    n_epochs = NUM_EPOCHS,
                    seq_len = BPTT_SEQUENCE_LENGTH,
                    lr = LEARNING_RATE,
                    batch_size = BATCH_SIZE,
                    vector_cache = VECTOR_CACHE,
                    objective = 'nce',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    model_type = "LSTM",
                    savepath = MODEL_SAVE_PATH,
                    wordvec_dim = WORDVEC_DIM,
                    wordvec_source = WORDVEC_SOURCE
                ):
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False
        self.savepath = savepath
        self.lr = lr
        self.datapath = datapath
        self.batch_size = batch_size
        self.bptt_len = seq_len
        self.n_epochs = n_epochs
        self.vector_cache = vector_cache
        self.objective_function = objective

        self.log_interval = log_interval
        self.model_type = model_type
        self.wordvec_source = wordvec_source
        self.wordvec_dim = wordvec_dim


    def load_data(self):

        print("Preparing Data Loaders")
        self.sentence_field = data.Field(
                            sequential = True,
                            use_vocab = True,
                            init_token = '<BOS>',
                            eos_token = '<EOS>',
                            fix_length = 100,
                            preprocessing = data.Pipeline(convert_token = preprocess), #function to preprocess if needed, already converted to lower, probably need to strip stuff
                            tensor_type = torch.LongTensor,
                            lower = True,
                            tokenize = 'spacy',
                        )
        if self.datapath is not None:
            print(self.vector_cache)
            print("Retrieving Data from file: {}...".format(self.datapath))
            self.raw_sentences = datasets.LanguageModelingDataset(self.datapath, self.sentence_field, newline_eos = False)
            print('done.')

#            self.raw_sentences = data.TabularDataset(
#                                path = self.datapath,
#                                format = 'csv',
#                                fields = [('text', self.sentence_field)]
#                            )
#
        else:
            print('Downloading Data Remotely....')
            self.raw_sentences = datasets.WikiText2.splits(self.sentence_field, root = 'data', train = 'wikitext-2/wiki.train.tokens', validation = None, test = None)[0]
            print('done.')


    def get_vectors(self):
        vecs = []
        if SAVED_VECTORS:
            print('Loading Vectors From Memory...')
            for source in self.wordvec_source:
                if source == 'GloVe':
                    glove = Vectors(name = 'glove.6B.{}d.txt'.format(self.wordvec_dim), cache = self.vector_cache)
                    vecs.append(glove)
                if source == 'charLevel':
                    charVec = Vectors(name = 'charNgram.txt',cache = self.vector_cache)
                    vecs.append(charVec)
        else:
            print('Downloading Vectors...')
            for source in self.wordvec_source:
                if source == 'GloVe':
                    glove = GloVe(name = '6B', dim = self.wordvec_dim, cache = self.vector_cache)
                    vecs.append(glove)
                if source == 'charLevel':
                    charVec = CharNGram(cache = self.vector_cache)
                    vecs.append(charVec)
        print('Building Vocab...')
        self.sentence_field.build_vocab(self.raw_sentences, vectors = vecs)
        print('Done.')


    def get_iterator(self):
        print('Getting Batches...')
        if self.cuda:
            self.batch_iterator = data.BPTTIterator(self.raw_sentences, sort_key = None, bptt_len = self.bptt_len, batch_size = self.batch_size)
            self.batch_iterator.repeat = False
        else:
            self.batch_iterator = data.BPTTIterator(self.raw_sentences, sort_key = None, bptt_len = self.bptt_len,  batch_size = self.batch_size, device = -1)
            self.batch_iterator.repeat = False

        print("Done.")

    def get_model(self):
        print('Initializing Model parameters...')
        self.ntokens = self.sentence_field.vocab.__len__()
        if self.model_type == "LSTM":
            self.model = LangModel(vocab_size = self.ntokens, pretrained_vecs = self.sentence_field.vocab.vectors)
            if self.objective_function == 'crossentropy':
                self.objective = CrossEntropyLoss()
            elif self.objective_function == 'nce':
                freqs = get_freqs(self.sentence_field.vocab)
                noise = build_unigram_noise(freqs)
                self.objective = NCELoss(self.ntokens, self.model.hidden_size, noise)


    def repackage_hidden(self, h):
        #Erasing hidden state history
        if type(h) == Variable:
            return Variable(h.data.type(torch.FloatTensor))
        else:
            return tuple(self.repackage_hidden(v) for v in h)


    def train_step(self, optimizer, model, start_time):
        hidden = self.model.init_hidden(self.batch_size)
        total_loss = 0
        for i, batch in enumerate(self.batch_iterator):
            optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)
            data, targets = batch.text, batch.target.view(-1)
            output, hidden = model(data, hidden)
            predictions = output.view(-1, self.ntokens)
            loss = self.objective(predictions, targets)
            loss.backward()
            total_loss += loss.data
            optimizer.step()
            if i % self.log_interval == 0:
                current_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                #OTHER STUFFFFFF
                total_loss = 0
                print('At time: {elapsed} loss is {current_loss}'.format(elapsed=elapsed, current_loss = current_loss))

    def train(self):
        print('Begin Training...')
        self.load_data()
        self.get_vectors()
        self.get_iterator()
        self.get_model()
        self.model.train()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = Adam(parameters)
        start_time = time.time()
        for epoch in range(self.n_epochs):
            print('Finished {} epochs...'.format(epoch))
            self.train_step(optimizer, self.model, start_time)

    def save_model(self, savepath):
        self.model.save_state_dict(savepath)

if __name__ == '__main__':

    trainer = TrainLangModel()
    trainer.train()
    trainer.save_model(trainer.savepath)


'''
print("DATA")
print(type(data.data), type(targets.data))
print(data.data.shape, targets.data.shape)
print("HIDDEN")
print([type(hidden[i].data) for i in range(2)])
print([vec.data.shape for vec in hidden])
print('PREDICTIONS')
print(type(output))
print(output.data.shape)
'''
