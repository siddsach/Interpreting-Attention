import torch.nn as nn
import torch
from torch.autograd import Variable

class VanillaRNN(nn.Module):

    def __init__(
            self,
            vocab_size,
            cuda,
            vectors,
            pretrained_rnn = None,
            model_type = 'LSTM',
            input_size = 300,
            hidden_size = 4096,
            num_layers = 1,
            rnn_dropout = 0.5,
            linear_dropout = None,
            tie_weights = False,
            init_range = 0.1,
            num_classes = 2,
            bidirectional = False,
            train_word_vecs = False
        ):

        super(VanillaRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_size) # this creates a layer

        self.model = getattr(nn, model_type)(input_size,
                                            hidden_size,
                                            num_layers,
                                            dropout = rnn_dropout,
                                            bidirectional = bidirectional,
                                            batch_first = True)

        if pretrained_rnn is not None:
            self.init_rnn(pretrained_rnn)

        if linear_dropout is not None:
            self.drop = nn.Dropout(p = linear_dropout)
        else:
            self.drop = lambda x: x

        self.decode_dim = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(self.decode_dim, num_classes)
        self.normalize = nn.Softmax()

        self.init_weights(init_range)

        self.model_type = model_type
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cuda = cuda
        self.num_classes = num_classes
        self.train_word_vecs = train_word_vecs

        print("EMBED")
        print(self.embed)
        print(vectors.shape)
        self.init_embedding(vectors)

    def init_embedding(self, pretrained_embeddings):
        self.embed.weight.data.copy_(pretrained_embeddings)# this provides the values
        if not self.train_word_vecs:
            self.embed.weight.requires_grad = False

    def init_rnn(self, pretrained_rnn):
        try:
            self.model.load_state_dict(pretrained_rnn)
        except:
            print("ERROR LOADING RNN WEIGHTS. PROCEEDING WITHOUT PRETRAINED_WEIGHTS")



    def init_weights(self, init_range):
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        num_states = 1
        if self.model_type == 'LSTM':
            num_states = 2
        num_directions = 1
        if self.bidirectional:
            num_directions = 2
        if self.cuda:
            return (Variable(torch.zeros(self.num_layers * num_directions,
                        batch_size, self.hidden_size).cuda(), requires_grad=False)
                        .type(torch.LongTensor) for i in range(num_states))
        else:
            return (Variable(torch.zeros(self.num_layers * num_directions,
                        batch_size, self.hidden_size), requires_grad=False)
                        .type(torch.LongTensor) for i in range(num_states))

    def forward(self, inp, h, lengths = None):
        vectors = self.embed(inp)
        packed_vecs = torch.nn.utils.rnn.pack_padded_sequence(vectors, list(lengths), batch_first = True)
        out, h = self.model(packed_vecs, h)

        print(h.data.shape)
        if self.bidirectional:
            hiddens = torch.cat((h[0][0], h[0][1]), 1)

        proj = self.linear(hiddens)
        predictions = self.normalize(proj)
        return predictions, h, None

class SelfAttentiveRNN(VanillaRNN):

    def __init__(
            self,
            attention_dim,
            mlp_hidden,
            train_hidden = True,
            **kwargs
        ):

        super(SelfAttentiveRNN, self).__init__(**kwargs)

        if not train_hidden:
            self.model.weight.requires_grad = False

        self.attention_dim = attention_dim

        if self.bidirectional:
            self.input_hidden_size = self.hidden_size * 2
        else:
            self.input_hidden_size = self.hidden_size

        # ATTENTION LAYERS
        self.W1 = nn.Linear(self.input_hidden_size, attention_dim, bias=False)
        self.W2 = nn.Linear(self.attention_dim, 1, bias=False )

        # MLP AND DECODER TO OUTPUT
        self.MLP = nn.Linear(self.input_hidden_size, mlp_hidden)
        self.decoder = nn.Linear(mlp_hidden, self.num_classes)

    def forward(self, inp, h, lengths = None):

        #EMBED, APPLY RNN
        vectors = self.embed(inp)
        packed_vecs = torch.nn.utils.rnn.pack_padded_sequence(vectors, list(lengths), batch_first = True)
        out, h = self.model(packed_vecs, h)
        out, lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first = True)

        # GET SELF-ATTENTION WEIGHTS
        s1 = self.W1(out)
        s2 = self.W2(nn.functional.tanh(s1))
        A = torch.squeeze(nn.functional.softmax(s2))

        #GET EMBEDDING MATRIX GIVEN ATTENTION WEIGHTS
        M = torch.sum(A.unsqueeze(2).expand_as(out) * out, 1)


        # DECODING ATTENTION EMBEDDED MATRICES TO OUTPUT
        MLPhidden = self.MLP(M)
        decoded = self.normalize(self.decoder(nn.functional.relu(MLPhidden)))

        return decoded, h, A
