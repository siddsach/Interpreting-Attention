import torch
import torch.nn as nn
from torch.autograd import Variable


#HYPERPARAMS
ENCODE_HIDDEN_DIM = 4096
DECODE_HIDDEN_DIM = 2048
WORD_VEC_SIZE = 300
DROPOUT = 0.5
BATCH_SIZE = 128
N_EPOCHS = 400
ARCHITECTURE = 'LSTM'
NUM_LAYERS = 1
BIDIRECTIONAL = False
USE_CUDA = torch.cuda.is_available()

class Encoder(nn.Module):


        def __init__(
                self,
                vocab_size,
                hidden_size = ENCODE_HIDDEN_DIM,
                input_size = WORD_VEC_SIZE,
                rnn_dropout = DROPOUT,
                num_layers = NUM_LAYERS,
                model_type = 'ARCHITECTURE',
                bid = BIDIRECTIONAL,
                use_cuda = USE_CUDA
            ):
            super(Encoder, self).__init__()

            self.embed = nn.Embedding(vocab_size, input_size)
            #define NN
            self.model = getattr(nn, model_type)(
                                        input_size,
                                        hidden_size,
                                        num_layers,
                                        dropout = rnn_dropout,
                                        bidirectional = bid
                                    )


            self.hidden_dim = hidden_size
            self.use_cuda = use_cuda

        def init_hidden(self, batch_size):
            num_states = 1
            if self.model_type == 'LSTM':
                num_states = 2
            num_directions = 1
            if self.bidirectional:
                num_directions = 2
            return (Variable(torch.zeros(self.num_layers * num_directions,
                            batch_size, self.hidden_size), requires_grad=False)
                            .type(torch.LongTensor) for i in range(num_states))

        def forward(self, inp, hidden):
            out, hidden = self.enc_lstm(inp, hidden)
            if self.use_cuda:
                return out.cuda(), hidden.cuda()
            else:
                return out, hidden

class Decoder(nn.Module):

    def __init__(self,
            vocab_size,
            hidden_size = DECODE_HIDDEN_DIM,
            output_size = ENCODE_HIDDEN_DIM,
            dropout = DROPOUT,
            num_layers = NUM_LAYERS,
            model_type = ARCHITECTURE,
            hidden_state = None,
            use_cuda = USE_CUDA
        ):

            super(Decoder, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size

            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.model = getattr(nn, model_type)(
                                        hidden_size,
                                        hidden_size,
                                        num_layers,
                                        dropout = dropout,
                                    )

            self.linear2vocab = nn.Linear(hidden_size, vocab_size)
            self.activation = nn.LogSoftmax()
            self.use_cuda = use_cuda


    def init_hidden(self, batch_size):
        num_states = 1
        if self.model_type == 'LSTM':
            num_states = 2
        return (Variable(torch.zeros(self.num_layers,
                        batch_size, self.hidden_size), requires_grad=False)
                        .type(torch.LongTensor) for i in range(num_states))

    def forward(self, inp, hidden):
        input_embeddings = self.embed(inp)
        out, hidden = self.enc_lstm(input_embeddings, hidden)
        scores = self.activation(self.linear2vocab(hidden))
        if self.use_cuda:
            return scores.cuda()
        else:
            return scores

class AttentionDecoder(Decoder):

    def __init__(self,
            attention_dim,
            **kwargs
            ):

        super(AttentionDecoder, self).__init__()
        self.attention_dim = attention_dim

        self.attn = nn.Linear(self.output_size, attention_dim)

    def forward(self, inp, encoder_outputs):

        #hidden_all (seq_len, bsz, hidden_size)

        attn_weights = self.attn(encoder_outputs).unsqueeze(0)
        norm_hidden = torch.mm(attn_weights, encoder_outputs) #Output of dim (1, bsz, hidden_size)

        input_embeddings = self.model(inp)

        out, hidden = self.enc_lstm(input_embeddings, norm_hidden)
        scores = self.activation(self.linear2vocab(hidden))

        if self.use_cuda:
            return scores.cuda()
        else:
            return scores
