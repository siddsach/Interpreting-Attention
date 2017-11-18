from torchtext import data
from torchtext.vocab import Vectors, GloVe, CharNGram
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from .model import VanillaRNN, SelfAttentiveRNN
import time
import glob
import os

current_path = os.getcwd()

print("CURRENT PATH:{}".format(current_path))
root_path = current_path#[:len(current_path) - len('classifier/attention_rnn') + 1]

print("ROOT PATH:{}".format(root_path))

DATASET = 'IMDB'
IMDB_PATH = current_path + '/data/imdb/aclImdb'# 'sentence_subjectivity.csv' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
VECTOR_CACHE = root_path + '/vectors'
SAVED_VECTORS = True
NUM_EPOCHS = 10
LEARNING_RATE = 0.5
BATCH_SIZE = 5
LOG_INTERVAL = 5
WORD_VEC_DIM = 300
WORDVEC_SOURCE = ['GloVe'] #['GloVe']# charLevel']
SAVED_MODEL_PATH = 'saved_model.pt'
IMDB = True
HIDDEN_SIZE = 4096
PRETRAINED = root_path + '/trained_models/trained_rnn.pt'
MAX_LENGTH = 100

def sorter(example):
    return len(example.text)

class TrainClassifier:
    def __init__(
                    self,
                    num_classes = 2,
                    pretrained_modelpath = PRETRAINED,
                    datapath = DATASET,
                    n_epochs = NUM_EPOCHS,
                    lr = LEARNING_RATE,
                    batch_size = BATCH_SIZE,
                    vector_cache = VECTOR_CACHE,
                    objective = 'crossentropy',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    model_type = "LSTM",
                    attention_dim = 10, #None if not using attention
                    mlp_hidden = 100,
                    wordvec_dim = WORD_VEC_DIM,
                    wordvec_source = WORDVEC_SOURCE,
                    hidden_dim = HIDDEN_SIZE,
                    max_length = MAX_LENGTH
                ):
        if torch.cuda.is_available():
            print("Using CUDA!")
            self.cuda = True
        else:
            print("Not Using CUDA")
            self.cuda = False
        self.lr = lr

        self.datapath = datapath
        if datapath == 'IMDB':
            self.trainpath = IMDB_PATH + "/train"
            self.testpath = IMDB_PATH + "/test"

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.vector_cache = vector_cache

        if objective == 'crossentropy':
            self.objective = CrossEntropyLoss()

        #MODEL SPECS
        self.model_type = model_type
        self.attention_dim = attention_dim
        self.mlp_hidden = mlp_hidden
        self.num_classes = num_classes


        self.log_interval = log_interval
        self.wordvec_source = wordvec_source
        self.wordvec_dim = wordvec_dim
        self.hidden_dim = hidden_dim
        self.pretrained_modelpath = pretrained_modelpath
        self.max_length = max_length

        self.sentence_field = data.Field(
                            sequential = True,
                            use_vocab = True,
                            init_token = '<BOS>',
                            eos_token = '<EOS>',
                            fix_length = self.max_length,
                            include_lengths = True,
                            preprocessing = None, #function to preprocess if needed, already converted to lower, probably need to strip stuff
                            tensor_type = torch.LongTensor,
                            lower = True,
                            tokenize = 'spacy',
                            batch_first = True
                        )

        self.target_field = data.Field(sequential = False, batch_first = True)



    def get_data(self, path):

        print("Retrieving Data from file: {}...".format(path))

        fields = [('text', self.sentence_field), ('label', self.target_field)]
        examples = []

        for label in ['pos', 'neg']:
            for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                with open(fname, 'r') as f:
                    text = f.readline()
                examples.append(data.Example.fromlist([text, label], fields))

        return data.Dataset(examples, fields)

    def load_data(self):

        fields = [('text', self.sentence_field),
                  ('label', self.target_field)]

        if self.trainpath is not None:
            self.train_data = self.get_data(self.trainpath)

        if self.testpath is not None:
            self.test_data = self.get_data(self.testpath)


        else:
            self.sentences = data.TabularDataset(
                            path = self.datapath,
                            format = 'csv',
                            fields = fields
                        )

    def get_vectors(self):
        vecs = []
        if SAVED_VECTORS:
            print('Loading Vectors From Memory...')
            print(self.wordvec_source)
            for source in self.wordvec_source:
                if source == 'GloVe':
                    print('Getting GloVe Vectors with {} dims'.format(self.wordvec_dim))
                    glove = Vectors(name = 'glove.6B.{}d.txt'.format(self.wordvec_dim), cache = self.vector_cache)
                    vecs.append(glove)
                if source == 'charLevel':
                    print('Getting charLevel Vectors')
                    charVec = Vectors(name = 'charNgram.txt', cache = self.vector_cache)
                    vecs.append(charVec)
                if source == 'googlenews':
                    print('Getting google news vectors')
                    google = Vectors(name = 'googlenews.bin', cache = self.vector_cache)
                    vecs.append(google)
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
        self.sentence_field.build_vocab(self.train_data, vectors = vecs)
        self.target_field.build_vocab(self.train_data)
        print('Done.')



    def build_batches(self, dataset):
        print('Getting Batches...')
        if self.cuda:
            iterator_object = data.Iterator(dataset, sort_key = sorter,
                                                batch_size = self.batch_size,
                                                sort = True)
            iterator_object.repeat = False
        else:
            iterator_object = data.Iterator(dataset, sort_key = sorter,
                                                sort = True,batch_size = self.batch_size,
                                                device = -1)
            iterator_object.repeat = False

        print("Done.")
        return iterator_object

    def get_batches(self):
        if self.train_data is not None:
            self.train_iterator = self.build_batches(self.train_data)
        if self.test_data is not None:
            self.test_iterator = self.build_batches(self.test_data)

    def get_model(self, num_tokens = None):
        print('Building model...')

        self.ntokens = len(self.sentence_field.vocab)

        pretrained_model = None

        if self.pretrained_modelpath is not None:
            pretrained_model = torch.load(self.pretrained_modelpath)
            print('Using Pretrained RNN from path: {}'.format(self.pretrained_modelpath))

        if self.attention_dim is not None:
            print('Using Attention model with {} dimensions'.format(self.attention_dim))
            self.model = SelfAttentiveRNN(vocab_size = self.ntokens,
                                            cuda = self.cuda,
                                            num_classes = self.num_classes,
                                            vectors = self.sentence_field.vocab.vectors,
                                            pretrained_rnn = pretrained_model,
                                            attention_dim = self.attention_dim,
                                            mlp_hidden = self.mlp_hidden
                                        )

            #MAKING MATRIX TO SAVE ATTENTION WEIGHTS
            self.train_attns = torch.zeros(2, len(self.train_data), self.max_length)
            self.eval_attns = torch.zeros(2, len(self.test_data), self.max_length)
        else:
            print('Using Vanilla RNN with {} dimensions'.format(self.hidden_dim))
            self.model = VanillaRNN(vocab_size = self.ntokens,
                                    cuda = self.cuda,
                                    num_classes = self.num_classes,
                                    hidden_size = self.hidden_dim,
                                    vectors = self.sentence_field.vocab.vectors,
                                    pretrained_rnn = pretrained_model.model.model
                                )
        print('Done.')


    def repackage_hidden(self, h):
        '''Wraps hidden states in new Variables, to detach them from their history.'''
        if type(h) == Variable:
            if self.cuda:
                return Variable(h.data.type(torch.FloatTensor).cuda())
            else:
                return Variable(h.data.type(torch.FloatTensor))
        else:
            return tuple(self.repackage_hidden(v) for v in h)


    def evaluate(self):
        self.model.eval()
        hidden = self.model.init_hidden(self.batch_size)
        total_loss = 0
        i = 0
        for i, batch in enumerate(self.test_iterator):

            #CLEARING HISTORY
            hidden = self.repackage_hidden(hidden)

            #GETTING TENSORS
            data, targets = batch.text, batch.label.view(-1)
            targets = targets - 1 #NEED TO INDEX FROM ZERO
            data, lengths = data[0], data[1]

            #GETTING PREDICTIONS
            output, h, A = self.model(data, hidden, lengths = lengths)
            predictions = output.view(-1, self.num_classes)

            #SAVING ATTENTION WEIGHTS
            self.save_attns(i, data, A, "test")

            #CALCULATING LOSS
            loss = self.objective(predictions, targets)
            total_loss += loss.data
        return total_loss

    def train_step(self, optimizer, start_time):
        hidden = self.model.init_hidden(self.batch_size)
        total_loss = 0
        for i, batch in enumerate(self.train_iterator):
            #CLEARING HISTORY
            optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)

            #GETTING TENSORS
            data, targets = batch.text, batch.label.view(-1)
            targets = targets - 1 #NEED TO INDEX FROM ZERO
            data, lengths = data[0], data[1]

            #GETTING PREDICTIONS
            output, h, A = self.model(data, hidden, lengths = lengths)
            predictions = output.view(-1, self.num_classes)

            #SAVING ATTENTION WEIGHTS
            self.save_attns(i, data, A, 'train')

            #CALCULATING AND PROPAGATING LOSS
            loss = self.objective(predictions, targets)
            loss.backward()
            total_loss += loss.data
            optimizer.step()


            if i % self.log_interval == 0:
                current_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                total_loss = 0
                print('At time: {elapsed}\n loss is {current_loss}'.format(elapsed=elapsed, current_loss = current_loss[0]))

    def save_attns(self, i, text, attns, fold = 'train'):
        index = self.batch_size * i

        if fold == 'train':
            #SAVE TEXT
            self.train_attns[0, index: index + self.batch_size, :attns.size(1)] = text.data[:, :attns.size(1)]
            #SAVE ATTENTION WEIGHTS
            self.train_attns[1, index: index + self.batch_size, :attns.size(1)] = attns.data

        elif fold == 'test':
            #SAVE TEXT
            self.test_attns[0, index: index + self.batch_size, :attns.size(1)] = text.data[:, :attns.size(1)]
            #SAVE ATTENTION WEIGHTS
            self.test_attns[1, index: index + self.batch_size, :attns.size(1)] = attns.data


    def save_model(self, savepath):
        self.model.save_state_dict(savepath)

    def dump_attns(self, attn_path):
        if self.test_attns is not None:
            torch.save(self.test_attns, attn_path)

    def train(self):
        print("Building RNN Classifier...")
        self.load_data()
        self.get_vectors()
        self.get_batches()
        self.get_model()
        self.model.train()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = Adam(parameters)
        start_time = time.time()
        print('Begin Training...')
        for epoch in range(self.n_epochs):
            self.train_step(optimizer, start_time)

if __name__ == '__main__':
    trainer = TrainClassifier()
    trainer.train()
    print("Evaluation loss is: {}".format(trainer.evaluate()))
    trainer.save_model(trainer.savepath)


