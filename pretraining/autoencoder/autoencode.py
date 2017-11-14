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
                vectors = None,
                attention_dim = None,
                tune_wordvecs = False,
                hidden_size = ENCODE_HIDDEN_DIM,
                input_size = WORD_VEC_SIZE,
                rnn_dropout = DROPOUT,
                num_layers = NUM_LAYERS,
                model_type = ARCHITECTURE,
                bid = BIDIRECTIONAL,
                use_cuda = USE_CUDA
            ):
            super(Encoder, self).__init__()

            self.hidden_dim = hidden_size
            self.use_cuda = use_cuda
            self.tune_wordvecs = tune_wordvecs
            self.model_type = model_type
            self.bidirectional = bid

            self.embed = nn.Embedding(vocab_size, input_size)

            if vectors is not None:
                self.init_embedding(vectors)
            #define NN
            self.model = getattr(nn, model_type)(
                                        input_size,
                                        hidden_size,
                                        num_layers,
                                        dropout = rnn_dropout,
                                        bidirectional = bid
                                    )

            if attention_dim is not None:
                self.attention_dim = attention_dim

                self.W1 = nn.Linear(hidden_size, attention_dim)
                self.W2 = nn.Linear(attention_dim, 1)


        def init_embedding(self, pretrained_embeddings):
            self.embed.weight.data.copy_(pretrained_embeddings)# this provides the values
            if not self.tune_wordvecs:
                self.embed.weight.requires_grad = False

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
            embedded = self.embed(inp)
            out, hidden = self.enc_lstm(embedded, hidden)

            if self.attention_dim is None:
                if self.use_cuda:
                    return out.cuda(), hidden.cuda()
                else:
                    return out, hidden

            #GLOBAL ATTENTION WEIGHTING FOR AUTOENCODER
            S1 = nn.functional.tanh(self.W1(out))
            attn_weights = nn.functonal.softmax(self.W2(S1))

            weighted_hidden = torch.matmul(out, attn_weights.t())

            if self.use_cuda:
                return out.cuda(), weighted_hidden.cuda()
            else:
                return out, weighted_hidden


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
        inp = self.embed(inp)
        out, hidden = self.enc_lstm(inp, hidden)
        scores, hidden = self.activation(self.linear2vocab(hidden))
        if self.use_cuda:
            return scores.cuda(), hidden.cuda()
        else:
            return scores, hidden
