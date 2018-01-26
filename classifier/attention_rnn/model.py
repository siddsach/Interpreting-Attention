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
            model_type = 'LSTM',
            input_size = 300,
            hidden_size = 4096,
            num_layers = 1,
            dropout = 0.5,
            rnn_dropout = 0.0,
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
                                            dropout = rnn_dropout,
                                            bidirectional = bidirectional,
                                            batch_first = True
                                        )



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
            print('Not Tuning Word Vectors!')
            self.embed.weight.requires_grad = False

    def init_pretrained(self, pretrained, fix_pretrained = 2):
#        try:

        use = lambda key: ('model' in key or 'embed' in key)
        pretrained = {key: pretrained[key] for key in pretrained if use(key)}


        #LOAD PARAMS FROM PRETRAINED MODEL, IGNORING IRRELEVANT ONES
        self.load_state_dict(pretrained, strict = False)

        for key in pretrained.keys():
            if key in self.state_dict().keys():
                try:
                    assert self.state_dict()[key].equal(pretrained[key]), 'key not the same:{}'.format(key)
                except:
                    print('THIS:{}'.format(self.state_dict()[key]))
                    print('PRETRAINED:{}'.format(pretrained[key]))

        #OPTION TO FIX PRETRAINED PARAMETERS
        if fix_pretrained is not None:
            self.embed.weight.requires_grad = False
            for i in range(fix_pretrained):
                for param in self.model._parameters.keys():
                    if str(i) in param:
                        print('Keeping pretrained param:{} fixed'.format(param))
                        self.model._parameters[param].requires_grad = False


#        except:
#            print("ERROR LOADING RNN WEIGHTS. PROCEEDING WITHOUT PRETRAINED_WEIGHTS")



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

        vectors = self.drop(self.embed(inp))
        packed_vecs = torch.nn.utils.rnn.pack_padded_sequence(vectors, list(lengths), batch_first = True)
        out, hiddens = self.model(packed_vecs, self.hiddens)

        if self.bidirectional:
            hiddens = torch.cat((hiddens[0][0], hiddens[0][1]), 1)

        out = torch.nn.utils.rnn.pad_packed_sequence(out, list(lengths))

        #features = out[0][:, list(lengths - 1), :]

        features = hiddens[0][self.num_layers - 1, :, :].squeeze(0)

        features = self.drop(features)

        proj = self.linear(features)
        probs = self.normalize(proj)

        return probs, hiddens, None

class SelfAttentiveRNN(VanillaRNN):

    def __init__(
            self,
            attention_dim = 300,
            train_hidden = True,
            attn_type = 'similarity',
            tune_attn = True,
            **kwargs
        ):

        super(SelfAttentiveRNN, self).__init__(**kwargs)

        if not train_hidden:
            self.model.weight.requires_grad = False

        if self.bidirectional:
            self.input_hidden_size = self.hidden_size * 2
        else:
            self.input_hidden_size = self.hidden_size

        assert attn_type == 'MLP' or attn_type == 'similarity', "ATTENTION TYPE MUST BE MLP OR similarity"

        self.attn_type = attn_type

        if self.attn_type == 'MLP':

            # ATTENTION LAYERS
            self.W1 = nn.Linear(self.input_hidden_size, attention_dim, bias=False)
            self.W2 = nn.Linear(attention_dim, 1, bias=False )


        elif self.attn_type == 'similarity':
            self.W = nn.Parameter(torch.randn(self.input_hidden_size, self.input_hidden_size))
            if not tune_attn:
                self.W.requires_grad = False
                nn.init.eye(self.W)


        # MLP AND DECODER TO OUTPUT
        self.decoder = nn.Linear(self.input_hidden_size, self.num_classes)

    def forward(self, inp, lengths = None):

        #EMBED, APPLY RNN
        vectors = self.embed(inp)
        packed_vecs = torch.nn.utils.rnn.pack_padded_sequence(vectors, list(lengths), batch_first = True)
        out, h = self.model(packed_vecs, self.hiddens)
        out, lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first = True)

        M = None
        A = None
        if self.attn_type == 'MLP':
            # GET SELF-ATTENTION WEIGHTS
            s1 = self.W1(out)
            s2 = self.W2(nn.functional.tanh(s1))
            A = torch.squeeze(nn.functional.softmax(s2))

        elif self.attn_type == 'similarity':
            #GET HIDDEN STATES
            last_hiddens = h[0][self.num_layers - 1, :, :]

            # GET ATTENTION WEIGHTS
            # A = (all_hiddens x W x last_hidden)
            attn_params = self.W.unsqueeze(0).expand(self.batch_size, self.W.size(0), self.W.size(1))
            weighted_seq = torch.bmm(out, attn_params)
            batched_last_hiddens = last_hiddens.unsqueeze(2)
            A = torch.bmm(weighted_seq, batched_last_hiddens).squeeze(2)
            A = nn.functional.softmax(A, dim = 1)

        #GET EMBEDDING MATRIX GIVEN ATTENTION WEIGHTS
        M = torch.sum(A.unsqueeze(2).expand_as(out) * out, 1)
        # DECODING ATTENTION EMBEDDED MATRICES TO OUTPUT
        decoded = self.normalize(self.decoder(M))

        return decoded, h, A


