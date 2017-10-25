from torchtext import data
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from model import LangModel
import time

RAW_TEXTDATA_PATH = '/Users/siddharth/flipsideML/ML-research/deep/semi-supervised_clf/data/more_sentences.csv' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
VECTOR_CACHE = 'vectors'
SAVED_VECTORS = False
NUM_EPOCHS = 10
BPTT_LENGTH = 10
LEARNING_RATE = 0.5
BATCH_SIZE = 64
LOG_INTERVAL = 10
BPTT_SEQUENCE_LENGTH = 2

class TrainLangModel:
    def __init__(
                    self,
                    datapath = RAW_TEXTDATA_PATH,
                    n_epochs = NUM_EPOCHS,
                    seq_len = BPTT_SEQUENCE_LENGTH,
                    lr = LEARNING_RATE,
                    batch_size = BATCH_SIZE,
                    vector_cache = VECTOR_CACHE,
                    objective = 'crossentropy',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    model_type = "LSTM"
                ):
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False
        self.lr = lr
        self.datapath = datapath
        self.batch_size = batch_size
        self.bptt_len = seq_len
        self.n_epochs = n_epochs
        self.vector_cache = vector_cache
        if objective == 'crossentropy':
            self.objective = CrossEntropyLoss()
        self.log_interval = log_interval
        self.model_type = model_type


    def load_data(self):

        print(self.vector_cache)
        print("Retrieving Data from file: {}...".format(self.datapath))
        self.sentence_field = data.Field(
                            sequential = True,
                            use_vocab = True,
                            init_token = '<BOS>',
                            eos_token = '<EOS>',
                            fix_length = 100,
                            preprocessing = None, #function to preprocess if needed, already converted to lower, probably need to strip stuff
                            lower = True,
                            tokenize = 'spacy',
                        )

        self.raw_sentences = data.TabularDataset(
                            path = self.datapath,
                            format = 'csv',
                            fields = [('text', self.sentence_field)]
                        )

        self.sentence_field.build_vocab(self.raw_sentences)

        if SAVED_VECTORS:
            self.sentence_field.vocab.vectors = torch.load(self.vector_cache)
        else:
            self.sentence_field.vocab.load_vectors('glove.6B.300d')
            #os.makedirs(self.vector_cache)
            torch.save(self.sentence_field.vocab.vectors, open(self.vector_cache, 'wb'))

        print("Done.")

        #POTENTIALLY USEFUL LINK IN CASE I HAVE TROUBLE HERE: https://github.com/pytorch/text/issues/70
    def get_iterator(self):
        print('Getting Batches...')
        if self.cuda:
            print('one')
            self.batch_iterator = data.BPTTIterator(self.raw_sentences, sort_key = None, bptt_len = self.bptt_len, batch_size = self.batch_size)
            self.batch_iterator.repeat = False
        else:
            print('two')
            self.batch_iterator = data.BPTTIterator(self.raw_sentences, sort_key = None, bptt_len = self.bptt_len,  batch_size = self.batch_size, device = -1)
            self.batch_iterator.repeat = False

        print("Done.")


    def repackage_hidden(self, h):
        '''Wraps hidden states in new Variables, to detach them from their history.'''
        if type(h) == Variable:
            return Variable(h.data.type(torch.LongTensor))
        else:
            return tuple(self.repackage_hidden(v) for v in h)


    def train_step(self, optimizer, model):
        print('first step')
        start_time = time.time()
        hidden = self.model.init_hidden(self.batch_size)
        total_loss = 0
        i = 0
        for batch in self.batch_iterator:
            print(i)
            print(batch)
            i += 1
            optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)
            data, targets = batch.text, batch.target.view(-1)
            print("DATA")
            print(type(data.data))
            print(data.data.shape)
            print("HIDDEN")
            print([type(hidden[i].data) for i in range(2)])
            print([vec.data.shape for vec in hidden])
            output, hidden = model(data, hidden)
            loss = self.objective(output, targets)
            loss.backward()
            total_loss += loss.data
            optimizer.step()
            if i % self.log_interval == 0:
                current_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                #OTHER STUFFFFFF
                print('At time: {elapsed} loss is {current_loss}'.format(elapsed=elapsed, current_loss = current_loss))

    def train(self):
        print('before train')
        self.get_iterator()
        ntokens = self.sentence_field.vocab.__len__()
        if self.model_type == "LSTM":
            self.model = LangModel(ntokens)
        optimizer = Adam(self.model.parameters())
        for epoch in range(self.n_epochs):
            self.batch_iterator
            self.train_step(optimizer, self.model)



