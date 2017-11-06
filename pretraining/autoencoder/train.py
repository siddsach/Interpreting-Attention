from torchtext import data
from utils import preprocess
import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from model import LangModel

RAW_TEXTDATA_PATH = '' #DATA MUST BE IN CSV FORMAT WITH ONE FIELD TITLED SENTENCES CONTANING ONE LINE PER SENTENCE
VECTOR_CACHE = None
NUM_EPOCHS = 40
MAX_LENGTH = 50
LEARNING_RATE = 0.5

class TrainAutoEncoder:
    def __init__(
                    self,
                    datapath = RAW_TEXTDATA_PATH,
                    n_epochs = NUM_EPOCHS,
                    seq_len = MAX_LENGTH,
                    lr = LEARNING_RATE
                    objective = CrossEntropyLoss
                    train = False
                ):
        if torch.cuda.is_available:
            self.cuda = True
        self.lr = lr
        self.load_data(datapath)
        self.get_iterator()
        self.model = LangModel
        self.objective = objective
        if train:
            self.train()

    def load_data(self, datapath):
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
                            path = datapath, ,
                            format = 'csv',
                            fields = [sentence_field]
                        )

        self.sentence_field.build_vocab(raw_sentences, labeled_sentences)

        if VECTOR_CACHE is not None:
            self.sentence_field.vocab.vectors = torch.load(VECTOR_CACHE)
        else:
            self.sentence_field.vocab.load_vectors('glove.6B.300d')

#POTENTIALLY USEFUL LINK IN CASE I HAVE TROUBLE HERE: https://github.com/pytorch/text/issues/70
    def get_iterator(self)
        self.raw_iter = data.BPTTIterator(self.raw_sentences, batch_size = BATCH_SIZE, bptt_len = MAX_LENGTH)


    def train_step(self):
        objective = self.objective()
        ntokens = self.sentence_field.vocab.__len__()
        model = self.model(ntokens)
        model.train()
        total_loss = 0
        start_time = time.time()
        iterator = self.get_iterator()
        for batch, targets in iterator:
            hidden = model.init_hidden(batch_size)
            model.zero_grad()
            output, hidden = model(data, hidden)
            model.zero_grad()
            loss = objective(output, targets)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.data
            if i % log_interval == 0:
                current_loss - total_loss[0]/ log_interval
                elapsed = time.time() - start_time
                #OTHER STUFFFFFF
                print('At time: {elapsed} loss is {current_loss}'.format(elapsed=elapsed, current_loss = current_loss))

    def train(self, n_epoch)
        for epoch in n_epoch:
            self.train()
