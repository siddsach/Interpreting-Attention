import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.Functional as F


#HYPERPARAMS
HIDDEN_DIM = 4096
WORD_VEC_SIZE = 300
DROPOUT = 0.5
BATCH_SIZE = 128
N_EPOCHS = 400
ARCHITECTURE = 'LSTM'
NUM_LAYERS = 1
BIDIRECTIONAL = False
USE_CUDA = torch.cuda.is_available()

class Encoder(nn.Module):
        
        super(Encoder, self).__init__()
        
        def __init__(
                self, 
                hidden_dim = hidden_dim, 
                input_size = word_vec_size, 
                dropout = dropout, 
                n_layers = num_layers, 
                bid = bidirectional
                hidden_state = None
            ):
        #define NN
        self.enc_lstm = nn.LSTM(
                            input_size = input_size, 
                            hidden_size = hidden_dim, 
                            num_layers = n_layers, 
                            dropout = dropout, 
                            bidirectional = bid
                        )


        self.hidden_dim = hidden_dim
        if hidden_state is not None:
            self.hidden = hidden_state
        else:
            self.hidden = self.init_hidden()
        
    def init_hidden(self):
        #INITIALIZING HIDDEN AND CELL STATE
        cell = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        hidden = autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        if use_cuda:
            return (cell.cuda(), hidden.cuda())
        else
            return (cell, hidden)

    def forward(self, inp, hidden):
        out, hidden = self.enc_lstm(inp, hidden)
        if use_cuda:
            return out.cuda(), hidden.cuda()
        else:
            return out, hidden

class Decoder(nn.Module):

    super(Decoder, self).__init__()

    def __init__(
            self, 
            hidden_dim = hidden_dim, 
            input_size = word_vec_size, 
            dropout = dropout, 
            n_layers = num_layers, 
            hidden_state = None
            vocab_size
        ):


        self.dec_lstm = nn.LSTM(
                            input_size = hidden_dim
                            hidden_size = hidden_dim,
                            num_layers = n_layers,
                            dropout = dropout,
                        )
        
        self.linear2vocab = nn.Linear(hidden_dim, vocab_size)
        self.activation = nn.LogSoftmax()


    def init_hidden(self):
        #INITIALIZING HIDDEN AND CELL STATE
        cell = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        hidden = autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        if use_cuda:
            return (cell.cuda(), hidden.cuda())
        else
            return (cell, hidden)

    def forward(self, inp, hidden):
        out, hidden = self.enc_lstm(inp, hidden)
        probs = self.linear2vocab(self.activation(hidden))
        if self.use_cuda:
            return probs.cuda()
        else:   
            return probs
