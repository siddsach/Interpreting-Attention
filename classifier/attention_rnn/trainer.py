from torchtext import data
from torchtext.vocab import Vectors
import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.autograd import Variable
from torch.optim import Adam, SGD
from model import VanillaRNN, SelfAttentiveRNN
import time
import glob
import os
from datetime import datetime

current_path = os.getcwd()

print("CURRENT PATH:{}".format(current_path))
root_path = current_path#[:len(current_path) - len('classifier/attention_rnn') + 1]

print("ROOT PATH:{}".format(root_path))

DATASET = 'IMDB'
IMDB_PATH = current_path + '/data/imdb/aclImdb'# 'sentence_subjectivity.csv' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
VECTOR_CACHE = root_path + '/vectors'
SAVED_VECTORS = True
NUM_EPOCHS = 40
LEARNING_RATE = 0.06
BATCH_SIZE = 32
LOG_INTERVAL = 5
WORD_VEC_DIM = 300
WORDVEC_SOURCE = ['GloVe']
#['GloVe']# charLevel']
SAVED_MODEL_PATH = None#'saved_model.pt'
IMDB = True
HIDDEN_SIZE = 300
PRETRAINED = None #root_path + '/trained_models/trained_rnn.pt'
MAX_LENGTH = 100
SAVE_CHECKPOINT = root_path + '/trained_models/classifier/'
USE_ATTENTION = True
ATTENTION_DIM = 350 if USE_ATTENTION else None
L2 = 0.0001
DROPOUT = 0.5
MLP_HIDDEN = 512
OPTIMIZER = 'adam'
CLIP = 0.5

MAX_DATA_LEN = 500
if torch.cuda.is_available():
    MAX_DATA_LEN = 5000


def sorter(example):
    return len(example.text)

class TrainClassifier:
    def __init__(
                    self,
                    num_classes = 2,
                    pretrained_modelpath = PRETRAINED,
                    checkpoint = None,
                    datapath = DATASET,
                    num_epochs = NUM_EPOCHS,
                    lr = LEARNING_RATE,
                    batch_size = BATCH_SIZE,
                    vector_cache = VECTOR_CACHE,
                    objective = 'crossentropy',
                    train = False,
                    log_interval = LOG_INTERVAL,
                    model_type = "LSTM",
                    attention_dim = ATTENTION_DIM, #None if not using attention
                    mlp_hidden = MLP_HIDDEN,
                    wordvec_dim = WORD_VEC_DIM,
                    wordvec_source = WORDVEC_SOURCE,
                    hidden_dim = HIDDEN_SIZE,
                    max_length = MAX_LENGTH,
                    use_cuda = True,
                    savepath = SAVE_CHECKPOINT,
                    optim = 'adam',
                    max_data_len = MAX_DATA_LEN,
                    dropout = DROPOUT,
                    clip = CLIP
                ):

        self.savepath = savepath

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA!")
            self.cuda = True
        else:
            print("Not Using CUDA")
            self.cuda = False

        self.lr = lr

        self.optim = optim

        self.datapath = datapath
        if datapath == 'IMDB':
            self.trainpath = IMDB_PATH + "/train"
            self.testpath = IMDB_PATH + "/test"
        self.max_data_len = max_data_len

        self.batch_size = batch_size
        self.n_epochs = num_epochs
        self.vector_cache = vector_cache

        if objective == 'crossentropy':
            self.objective = CrossEntropyLoss()

        elif objective == 'nllloss':
            self.objective = NLLLoss()

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
        self.checkpoint_path = checkpoint
        self.max_length = max_length
        self.optim = optim
        self.dropout = dropout
        self.clip = clip

        self.losses = torch.zeros(self.n_epochs)

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



    def get_data(self, path, max_len):

        print("Retrieving Data from file: {}...".format(path))

        fields = [('text', self.sentence_field), ('label', self.target_field)]
        examples = []

        for label in ['pos', 'neg']:
            c = 0
            for fname in glob.iglob(os.path.join(path, label, '*.txt')):

                if max_len is not None:
                    if c > max_len:
                        break

                with open(fname, 'r') as f:
                    text = f.readline()
                examples.append(data.Example.fromlist([text, label], fields))

                c += 1

        return data.Dataset(examples, fields)

    def load_data(self):

        fields = [('text', self.sentence_field),
                  ('label', self.target_field)]

        if self.trainpath is not None:
            self.train_data = self.get_data(self.trainpath, self.max_data_len)

        if self.testpath is not None:
            if self.max_data_len is not None:
                self.max_data_len = self.max_data_len / 4
            self.test_data = self.get_data(self.testpath, self.max_data_len)


        else:
            self.sentences = data.TabularDataset(
                            path = self.datapath,
                            format = 'csv',
                            fields = fields
                        )

    def get_vectors(self):
        vecs = []
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
        print('Building Vocab...')
        if len(vecs) > 0:
            self.sentence_field.build_vocab(self.train_data, vectors = vecs)
            self.target_field.build_vocab(self.train_data)
        else:
            self.sentence_field.build_vocab(self.train_data)
            self.target_field.build_vocab(self.train_data)



    def build_batches(self, dataset):
        print('Getting Batches...')
        if self.cuda:
            iterator_object = data.BucketIterator(dataset,
                                            sort_key = sorter,
                                            batch_size = self.batch_size,
                                            sort = True
                                        )
            iterator_object.repeat = False
        else:
            iterator_object = data.BucketIterator(dataset,
                                            sort_key = sorter,
                                            sort = True,
                                            batch_size = self.batch_size,
                                            device = -1
                                        )
            iterator_object.repeat = False

        print("Created Iterator with {num} batches".format(num = len(iterator_object)))
        return iterator_object

    def get_batches(self):
        if self.train_data is not None:
            self.train_iterator = self.build_batches(self.train_data)
        if self.test_data is not None:
            self.test_iterator = self.build_batches(self.test_data)

    def get_model(self, num_tokens = None):
        if self.checkpoint_path is None:
            print('Building model...')

            self.ntokens = len(self.sentence_field.vocab)

            pretrained_model = None

            if self.pretrained_modelpath is not None:
                pretrained_model = torch.load(self.pretrained_modelpath)
                print('Using Pretrained RNN from path: {}'.format(self.pretrained_modelpath))

            if self.attention_dim is not None:
                print('Using Attention model with {} dimensions'.format(self.attention_dim))
                self.model = SelfAttentiveRNN(vocab_size = self.ntokens,
                                                num_classes = self.num_classes,
                                                batch_size = self.batch_size,
                                                vectors = self.sentence_field.vocab.vectors,
                                                pretrained_rnn = pretrained_model,
                                                attention_dim = self.attention_dim,
                                                mlp_hidden = self.mlp_hidden,
                                                input_size = self.wordvec_dim,
                                                dropout = self.dropout
                                            )

                #MAKING MATRIX TO SAVE ATTENTION WEIGHTS
                self.train_attns = torch.zeros(2, len(self.train_data), self.max_length)
                self.eval_attns = torch.zeros(2, len(self.test_data), self.max_length)
            else:
                print('Using Vanilla RNN with {} dimensions'.format(self.hidden_dim))
                self.model = VanillaRNN(vocab_size = self.ntokens,
                                        num_classes = self.num_classes,
                                        batch_size = self.batch_size,
                                        hidden_size = self.hidden_dim,
                                        vectors = self.sentence_field.vocab.vectors,
                                        pretrained_rnn = pretrained_model,
                                        dropout = self.dropout,
                                        input_size = self.wordvec_dim
                                    )
            if self.cuda:
                self.model.cuda()
        else:
            print('Loading Model from checkpoint')
            self.model = torch.load(self.checkpoint_path)




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
        total_loss = 0
        i = 0
        for i, batch in enumerate(self.test_iterator):
            #GETTING TENSORS
            data, targets = batch.text, batch.label.view(-1)
            data, lengths = data[0], data[1]
            targets = targets - 1

            #CONVERTING TO CUDA IF ON NEEDED
            if self.cuda:
                data = data.cuda()
                targets = targets.cuda()
                lengths = lengths.cuda()

            if data.size(0) == self.batch_size:

                #GETTING PREDICTIONS
                output, h, A = self.model(data, lengths = lengths)
                predictions = output.view(-1, self.num_classes)

                if A is not None and False:
                    #SAVING ATTENTION WEIGHTS
                    self.save_attns(i, data, A, "test")

                #CALCULATING LOSS
                loss = self.objective(predictions, targets)

                total_loss += loss.data
            else:
                break
        self.eval_loss = total_loss[0] / i
        print('Done Evaluating: Achieved loss of {}'
                .format(self.eval_loss))

    def train_step(self, optimizer, start_time):
        total_loss = 0

        for i, batch in enumerate(self.train_iterator):
            #CLEARING HISTORY
            optimizer.zero_grad

            #GETTING TENSORS
            data, targets = batch.text, batch.label.view(-1)
            targets = targets - 1 #from zero to one
            data, lengths = data[0], data[1]

            #CONVERTING TO CUDA IF ON NEEDED
            if self.cuda:
                data = data.cuda()
                targets = targets.cuda()
                lengths = lengths.cuda()

            if data.size(0) == self.batch_size:
                #GETTING PREDICTIONS
                output, h, A = self.model(data, lengths = lengths)
                predictions = output.view(-1, self.num_classes)

                if A is not None and False:
                    #SAVING ATTENTION WEIGHTS
                    self.save_attns(i, data, A, 'train')

                #CALCULATING AND PROPAGATING LOSS
                loss = self.objective(predictions, targets)
                loss.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                total_loss += loss.data
                if self.optim in ['adam', 'SGD']:
                    optimizer.step()
                elif self.optim == 'vanilla_grad':
                    parameters = filter(lambda p: p.requires_grad, self.model.parameters())
                    for p in parameters:
                        p.data.add_(-self.lr, p.grad.data)


                if i % self.log_interval == 0:
                    current_loss = total_loss / self.log_interval
                    elapsed = time.time() - start_time
                    total_loss = 0
                    print('At time: {elapsed}\n loss is {current_loss}'.format(elapsed=elapsed, current_loss = current_loss[0]))

        return optimizer

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


    def save_checkpoint(self, optimizer, checkpointpath, name = None):
        state = {
                    'epoch': self.epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_valid_loss': self.eval_loss,
                    'optimizer': None if optimizer is None else optimizer.state_dict(),
                    'losses': self.losses
                }
        savepath = checkpointpath + ''.join(str(datetime.now()).split())
        if name is not None:
            savepath = checkpointpath + name

        torch.save(state, savepath)


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
        optimizer = None
        if self.optim == 'adam':
            optimizer = Adam(parameters, lr = self.lr)
        elif self.optim == 'SGD':
            optimizer = SGD(parameters, lr = self.lr, weight_decay = L2)

        start_time = time.time()
        print('Begin Training...')

        self.eval_loss = 100000
        not_better = 0
        self.best_eval_loss = 10000
        self.best_model = None
        for epoch in range(self.n_epochs):
            print("Completing Train Step...")
            optimizer = self.train_step(optimizer, start_time)
            print("Evaluating...")
            self.evaluate()
            self.losses[epoch] = self.eval_loss
            self.epoch = epoch
            if self.eval_loss > self.best_eval_loss:
                not_better += 1

                if not_better >= 5:
                    if self.optim == 'vanilla_grad':
                        #Annealing
                        self.lr /= 4
                elif not_better >= 20:
                    print('Model not improving. Stopping early with {}'
                           'loss at {} epochs.'.format(self.best_eval_loss, self.epoch))
                    break
            else:
                self.best_eval_loss = self.eval_loss
                self.best_model = self.model

        if self.savepath is not None:

            print("Saving Model Parameters and Results...")
            self.save_checkpoint(optimizer, self.savepath)

            print('Finished Training.')

if __name__ == '__main__':
    trainer = TrainClassifier()
    trainer.train()

