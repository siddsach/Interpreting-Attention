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
            linear_dropout = 0.2,
            tie_weights = False,
            init_range = 0.1,
            tune_wordvecs = False,
            drop_embed = True
        ):

        super(LangModel, self).__init__()
        #TRY NOT TYING WEIGHTS WITH FIXED WORD VECTORS

        self.embed = nn.Embedding(vocab_size, input_size) # this creates a layer

        self.model = getattr(nn, model_type)(input_size, hidden_size, num_layers, dropout = rnn_dropout)

        self.drop = nn.Dropout(linear_dropout)
        self.drop_embed = drop_embed

        self.decoder = decoder
        if decoder == 'softmax':
            self.linear = nn.Linear(hidden_size, vocab_size)
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
        if pretrained_embeddings is not None:
            # INIT EMBEDDING WITH PRE-TRAINED WORD VECTORS
            self.embed.weight.data.copy_(pretrained_embeddings)
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
        vectors = self.embed(inp)
        if self.drop_embed:
            vectors = self.drop(vectors)

        out, h = self.model(vectors, h)
        out = self.drop(out)

        if self.decoder == 'softmax':
            squeezed = out.view(out.size(0) * out.size(1) , out.size(2))
            predictions = self.linear(squeezed)


            return predictions, h

        else:
            predictions = out
            return predictions

