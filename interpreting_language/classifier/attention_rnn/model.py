import torch.nn as nn
import torch
from torch.autograd import Variable

class VanillaRNN(nn.Module):

    def __init__(
            self,
            vocab_size,
            vectors,
            batch_size,
            cuda,
            pretrained_rnn = None,
            model_type = 'LSTM',
            input_size = 300,
            hidden_size = 4096,
            num_layers = 1,
            dropout = 0.5,
            tie_weights = False,
            init_range = 0.1,
            num_classes = 2,
            bidirectional = False, #Cannot have bidirectional embeddings in a language model
            train_word_vecs = False
        ):

        super(VanillaRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_size) # this creates a layer
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        num_states = 2 if model_type == 'LSTM' else 1
        num_directions = 2 if bidirectional else 1

        self.batch_size = batch_size

        if cuda:

            self.hiddens  = tuple(nn.Parameter(torch.randn(num_layers, self.batch_size * num_directions, self.hidden_size)
                            .type(torch.cuda.FloatTensor), requires_grad=True) for i in range(num_states))
        else:
            self.hiddens  = tuple(nn.Parameter(torch.randn(num_layers, self.batch_size * num_directions, self.hidden_size)
                            .type(torch.FloatTensor), requires_grad=True) for i in range(num_states))

        self.hiddens = self.hiddens[0] if (num_states == 1) else self.hiddens

        self.drop = nn.Dropout(dropout)

        self.model = getattr(nn, model_type)(input_size,
                                            hidden_size,
                                            num_layers,
                                            dropout = dropout,
                                            bidirectional = bidirectional,
                                            batch_first = True
                                        )

        if pretrained_rnn is not None:
            self.init_rnn(pretrained_rnn)


        self.decode_dim = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(self.decode_dim, num_classes)
        self.normalize = nn.LogSoftmax()

        self.init_weights(init_range)

        self.model_type = model_type
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.train_word_vecs = train_word_vecs

        if vectors is not None:
            self.init_embedding(vectors)

    def init_embedding(self, pretrained_embeddings):
        self.embed.weight.data.copy_(pretrained_embeddings)# this provides the values
        if not self.train_word_vecs:
            self.embed.weight.requires_grad = False

    def init_rnn(self, pretrained_rnn):
        try:
            if pretrained_rnn is not None:
                self.model.load_state_dict(pretrained_rnn.model.model)
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

        return (Variable(torch.zeros(self.num_layers * num_directions,
                    batch_size, self.hidden_size), requires_grad=False)
                    .type(torch.LongTensor) for i in range(num_states))

    def forward(self, inp, lengths = None):

        vectors = self.embed(inp)
        print("VECTORS")
        print(vectors)

        packed_vecs = torch.nn.utils.rnn.pack_padded_sequence(vectors, list(lengths), batch_first = True)
        print("HIDDENS")
        print(self.hiddens)
        print(self.model)
        out, hiddens = self.model(packed_vecs, self.hiddens)

        if self.bidirectional:
            hiddens = torch.cat((hiddens[0][0], hiddens[0][1]), 1)

        out = torch.nn.utils.rnn.pad_packed_sequence(out, list(lengths))

        #features = out[0][:, list(lengths - 1), :]

        features = hiddens[0][self.num_layers - 1, :, :].squeeze(0)

        proj = self.linear(features)


        return proj, hiddens, None

class SelfAttentiveRNN(VanillaRNN):

    def __init__(
            self,
            attention_dim,
            mlp_hidden,
            train_hidden = True,
            attn_type = 'keyval',
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

        self.attn_type = attn_type

        if self.attn_type == 'MLP':

            # ATTENTION LAYERS
            self.W1 = nn.Linear(self.input_hidden_size, attention_dim, bias=False)
            self.W2 = nn.Linear(self.attention_dim, 1, bias=False )

            # MLP AND DECODER TO OUTPUT
            self.MLP = nn.Linear(self.input_hidden_size, mlp_hidden)
            self.decoder = nn.Linear(mlp_hidden, self.num_classes)

        elif self.attn_type == 'keyval':
            self.W = nn.Parameter(torch.randn(self.input_hidden_size, self.input_hidden_size))

    def forward(self, inp, lengths = None):

        #EMBED, APPLY RNN
        vectors = self.embed(inp)
        packed_vecs = torch.nn.utils.rnn.pack_padded_sequence(vectors, list(lengths), batch_first = True)
        print("VECS")
        print(packed_vecs.data.data.shape)
        print("HIDDENS")
        print(self.hiddens[0].data.shape)
        out, h = self.model(packed_vecs, self.hiddens)
        out, lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first = True)

        if self.attn_type == 'MLP':
            # GET SELF-ATTENTION WEIGHTS
            s1 = self.W1(out)
            s2 = self.W2(nn.functional.tanh(s1))
            A = torch.squeeze(nn.functional.softmax(s2))

            #GET EMBEDDING MATRIX GIVEN ATTENTION WEIGHTS
            M = torch.sum(A.unsqueeze(2).expand_as(out) * out, 1)

        elif self.attn_type == 'keyval':
            print("OUT")
            print(out)
            #GET HIDDENS
            print("W")
            print(self.W)
            print("HIDDENS")
            last_hiddens = out[:, -1, :]
            print(last_hiddens)

            #GET ATTENTION WEIGHTS
            weighted_seq = torch.mm(out, self.W)
            A = torch.mm(weighted_seq, last_hiddens)

            #GET ATTENTION WEIGHTED CONTEXT VECTOR
            M = torch.mm(A, out)

        # DECODING ATTENTION EMBEDDED MATRICES TO OUTPUT
        MLPhidden = self.MLP(M)
        decoded = self.normalize(self.decoder(nn.functional.relu(MLPhidden)))

        return decoded, h, A


