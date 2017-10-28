from torchtext import data
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from model import LangModel
import time
import pickle

RAW_TEXTDATA_PATH = '/Users/siddharth/flipsideML/ML-research/deep/semi-supervised_clf/data/more_sentences.csv' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
VECTOR_CACHE = 'vectors'
SAVED_VECTORS = False
NUM_EPOCHS = 10
BPTT_LENGTH = 3
LEARNING_RATE = 0.5
BATCH_SIZE =5
LOG_INTERVAL = 5
BPTT_SEQUENCE_LENGTH = 2
WORD_VEC_DIM = 300

'''
CREATING A POSTPROCESSING FUNCTION TO TURN SEQUENCES OF
INDEXES FOR EACH WORD INTO WORD VECTORS
'''
def convert_token(x, vocab, train):
    out = torch.zeros(len(x), WORD_VEC_DIM)
    if train:
        c = 0
        for i, word in enumerate(x):
            out[i] = vocab.vectors[word]
            c+= 1
            if c > 128:
                break
        print('FINISHED TURNING INTO VECTORS')
        print(type(out))
        print(out.shape)
        pickle.dump(out, open('weird_tensor.p', 'wb'))
        out = [torch.FloatTensor(out) for el in out]
        print('FINISHED CONVERTING')
        return out
        '''
        out = arr
        for i, example in enumerate(arr):
            new_example = len(example) * [None]
            for j, word in enumerate(example):
                vec = vocab.vectors[word]
                new_example[j] = vec
            out[i] = new_example
        print('Done preprocessing')
        return out
        '''

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
                            #postprocessing = data.Pipeline(convert_token = convert_token),
                            tensor_type = torch.LongTensor,
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
            self.sentence_field.vocab.load_vectors('fasttext.simple.300d')
            #os.makedirs(self.vector_cache)
            torch.save(self.sentence_field.vocab.vectors, open(self.vector_cache, 'wb'))

        print("Done.")

        #POTENTIALLY USEFUL LINK IN CASE I HAVE TROUBLE HERE: https://github.com/pytorch/text/issues/70


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

        self.ntokens = self.sentence_field.vocab.__len__()
        if self.model_type == "LSTM":
            self.model = LangModel(self.ntokens)
        self.model.init_embedding(self.sentence_field.vocab.vectors)

    def repackage_hidden(self, h):
        '''Wraps hidden states in new Variables, to detach them from their history.'''
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
        self.get_iterator()
        self.get_model()
        optimizer = Adam(self.model.parameters())
        start_time = time.time()
        for epoch in range(self.n_epochs):
            self.train_step(optimizer, self.model, start_time)


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
