from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe, CharNGram
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from model import VanillaRNN, SelfAttentiveRNN
import time

RAW_TEXTDATA_PATH = None # 'sentence_subjectivity.csv' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
VECTOR_CACHE = '/Users/siddharth/flipsideML/ML-research/deep/semi-supervised_clf/vectors'
SAVED_VECTORS = True
NUM_EPOCHS = 10
LEARNING_RATE = 0.5
BATCH_SIZE = 5
LOG_INTERVAL = 5
WORD_VEC_DIM = 300
WORDVEC_SOURCE = ['GloVe']# charLevel']
SAVED_MODEL_PATH = 'saved_model.pt'


class TrainClassifier:
    def __init__(
                    self,
                    num_classes = 2,
                    datapath = RAW_TEXTDATA_PATH,
                    n_epochs = NUM_EPOCHS,
                    lr = LEARNING_RATE,
                    batch_size = BATCH_SIZE,
                    vector_cache = VECTOR_CACHE,
                    objective = 'crossentropy',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    model_type = "LSTM",
                    attention_dim = 2, #None if not using attention
                    mlp_hidden = 100,
                    num_aspects = None,
                    word_vec_dim = WORD_VEC_DIM,
                    wordvec_source = WORDVEC_SOURCE
                ):
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False
        self.lr = lr
        self.datapath = datapath
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.vector_cache = vector_cache
        self.num_aspects = num_aspects
        if objective == 'crossentropy':
            self.objective = CrossEntropyLoss()
        self.log_interval = log_interval
        self.model_type = model_type
        self.num_classes = num_classes
        self.attention_dim = attention_dim
        self.mlp_hidden = mlp_hidden
        self.num_classes = num_classes
        self.wordvec_source = wordvec_source
        self.wordvec_dim = word_vec_dim

        self.sentence_field = data.Field(
                            sequential = True,
                            use_vocab = True,
                            init_token = '<BOS>',
                            eos_token = '<EOS>',
                            fix_length = 100,
                            preprocessing = None, #function to preprocess if needed, already converted to lower, probably need to strip stuff
                            tensor_type = torch.LongTensor,
                            lower = True,
                            tokenize = 'spacy',
                        )

        self.target_field = data.Field(sequential = False)


    def load_data(self):

        fields = [('text', self.sentence_field),
                  ('label', self.target_field)]

        if self.datapath is not None:
            print(self.vector_cache)
            print("Retrieving Data from file: {}...".format(self.datapath))

            self.sentences = data.TabularDataset(
                                path = self.datapath,
                                format = 'csv',
                                fields = fields
                            )
        else:
            print('Downloading IMDB data...')
            self.sentences = datasets.IMDB.splits(text_field = self.sentence_field,
                                            label_field = self.target_field,
                                            root = 'data', test = None)[0]
            self.target_field.build_vocab()

        print("Done.")

        #POTENTIALLY USEFUL LINK IN CASE I HAVE TROUBLE HERE: https://github.com/pytorch/text/issues/70


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
        self.sentence_field.build_vocab(self.sentences, vectors = vecs)
        print('Done.')



    def get_batches(self):
        print('Getting Batches...')
        if self.cuda:
            self.batch_iterator = data.BucketIterator(self.sentences, sort_key = None, batch_size = self.batch_size)
            self.batch_iterator.repeat = False
        else:
            self.batch_iterator = data.BucketIterator(self.sentences, sort_key = None, batch_size = self.batch_size, device = -1)
            self.batch_iterator.repeat = False

        print("Done.")

    def get_model(self, num_tokens = None):
        print('Building model...')
        if num_tokens is None:
            self.ntokens = len(self.sentence_field.vocab)
            if self.attention_dim is not None:
                print('Using Attention model with {} dimensions'.format(self.attention_dim))
                self.model = SelfAttentiveRNN(vocab_size = self.ntokens,
                                                cuda = self.cuda,
                                                num_classes = self.num_classes,
                                               vectors = self.sentence_field.vocab.vectors,
                                                attention_dim = self.attention_dim,
                                                mlp_hidden = self.mlp_hidden,
                                                num_aspects = self.num_aspects
                                        )
            else:
                print('Using Vanilla RNN with {} dimensions')
                self.model = VanillaRNN(vocab_size = self.ntokens,
                                                cuda = self.cuda,
                                                num_classes = self.num_classes,
                                                vectors = self.sentence_field.vocab.vectors,
                                     )
        print('Done.')


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
            data, targets = batch.text, batch.label.view(-1)
            output, h, penal, weights = model(data, hidden)
            print('OUTPUT')
            print(type(output))
            print(output.data.shape)
            print('WEIGHTS')
            print(type(weights))
            print(weights[0].data.shape)
            predictions = output.view(-1, self.num_classes)
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

    def save_model(self, savepath):
        self.model.save_state_dict(savepath)


    def train(self):
        print('Begin Training...')
        self.load_data()
        self.get_vectors()
        self.get_batches()
        self.get_model()
        self.model.train()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = Adam(parameters)
        start_time = time.time()
        for epoch in range(self.n_epochs):
            self.train_step(optimizer, self.model, start_time)

if __name__ == '__main__':
    trainer = TrainClassifier()
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
