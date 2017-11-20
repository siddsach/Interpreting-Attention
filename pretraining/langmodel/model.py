import torch.nn as nn
import torch
from torch.autograd import Variable

class LangModel(nn.Module):

    def __init__(
            self,
            vocab_size,
            pretrained_vecs,
            decoder = 'softmax',
            model_type = 'LSTM',
            input_size = 300,
            hidden_size = 4096,
            num_layers = 2,
            rnn_dropout = 0.2,
            linear_dropout = 0.22,
            tie_weights = False,
            init_range = 0.1,
            tune_wordvecs = False
        ):

        super(LangModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_size) # this creates a layer

        self.model = getattr(nn, model_type)(input_size, hidden_size, num_layers, dropout = rnn_dropout)

        self.drop = nn.Dropout(linear_dropout)
        self.decoder = decoder
        if decoder == 'softmax':
            self.linear = nn.Linear(hidden_size, vocab_size)
            self.normalize = nn.Softmax()
            self.init_weights(init_range)


        if tie_weights:
            assert hidden_size == input_size, "If you tie weights you gott have the same embedding and hidden size, stupid!"
            self.decoder.weight = self.encoder.weight


        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tune_wordvecs = tune_wordvecs

        self.init_embedding(pretrained_vecs)

    def init_embedding(self, pretrained_embeddings):
        self.embed.weight.data.copy_(pretrained_embeddings)# this provides the values
        if not self.tune_wordvecs:
            self.embed.weight.requires_grad = False

    def init_weights(self, init_range):
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        num_states = 1
        if self.model_type == 'LSTM':
            num_states = 2

        return (Variable(torch.zeros(self.num_layers, batch_size,
                        self.hidden_size), requires_grad=False)
                        .type(torch.LongTensor) for i in range(num_states))


    def forward(self, inp, h):
        vectors = self.drop(self.embed(inp))
        out, h = self.model(vectors, h)
        out = self.drop(out)

        if self.decoder == 'softmax':
            squeezed = out.view(out.size(0) * out.size(1) , out.size(2))
            predictions = self.normalize(self.linear(squeezed))
        else:
            predictions = out
        return predictions, h
