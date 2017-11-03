import torch.nn as nn
import torch
from torch.autograd import Variable

class VanillaRNN(nn.Module):

    def __init__(
            self,
            vocab_size,
            cuda,
            vectors,
            model_type = 'LSTM',
            input_size = 300,
            hidden_size = 4096,
            num_layers = 1,
            rnn_dropout = 0.5,
            linear_dropout = 0.5,
            tie_weights = False,
            init_range = 0.1,
            num_classes = 2,
            bidirectional = True,
            train_word_vecs = False
        ):

        super(VanillaRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_size) # this creates a layer

        self.model = getattr(nn, model_type)(input_size, hidden_size, num_layers, dropout = rnn_dropout, bidirectional = bidirectional)

        self.drop = nn.Dropout(linear_dropout)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.normalize = nn.Softmax()

        self.init_weights(init_range)

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cuda = cuda
        self.num_classes = num_classes
        self.train_word_vecs = train_word_vecs
        self.init_embedding(vectors)

    def init_embedding(self, pretrained_embeddings):
        self.embed.weight.data.copy_(pretrained_embeddings)# this provides the values
        if not self.train_word_vecs:
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
        out, h = self.model(vectors, h)
        out = self.drop(out)

        predictions = self.normalize(self.linear(out))
        return predictions, h, None, None

class SelfAttentiveRNN(VanillaRNN):

    def __init__(
            self,
            attention_dim,
            mlp_hidden,
            train_hidden = True,
            num_aspects = None, #None IF WE ONLY CONSIDER ONE ASPECT, IGNORE THIS ARG
            **kwargs
        ):

        super(SelfAttentiveRNN, self).__init__(**kwargs)

        if not train_hidden:
            self.model.weight.requires_grad = False

        self.attention_dim = attention_dim


        if num_aspects is not None:
            self.num_aspects = num_aspects
        else:
            self.num_aspects = 1

        if self.bidirectional:
            self.input_hidden_size = self.hidden_size * 2
        else:
            self.input_hidden_size = self.hidden_size

        # ATTENTION LAYERS
        self.W1 = nn.Linear(self.input_hidden_size, attention_dim, bias=False)
        self.W2 = nn.Linear(attention_dim, num_aspects, bias=False )

        # MLP AND DECODER TO OUTPUT
        self.MLP = nn.Linear(num_aspects * self.input_hidden_size, mlp_hidden)
        self.decoder = nn.Linear(mlp_hidden, self.num_classes)

    def forward(self, inp, h, lens):
        print('one')
        rnn_input = self.embed(inp)
        print('two')

        #rnn_input = torch.nn.utils.rnn.pack_padded_sequence( emb, list( len_li.data ), batch_first=True )
        output, h = self.rnn(rnn_input , h)
        print('three')

        #depacked_output, lens = torch.nn.utils.rnn.pad_packed_sequence( output, batch_first=True )

        Batched_Attentions = Variable(torch.zeros(input.size(0), self.num_aspects * self.input_hidden_size))

        if self.num_aspects > 1:
            penal = Variable(torch.zeros(1))
            I = Variable(torch.eye(self.num_aspects))

        if self.cuda:
            Batched_Attentions = Batched_Attentions.cuda()
            if self.num_aspects > 1:
                penal = penal.cuda()
                I = I.cuda()

        weights = {}


        print('finished sequence component')
        # ATTENTION CALCULATIONS
        for i in range( input.size( 0 ) ):

            #GETTING HIDDEN STATES FOR iTH EXAMPLE IN BATCH
            H = output[i, :lens[ i ], :]

            # GET SELF-ATTENTION WEIGHTS FOR THIS HIDDEN WEIGHT
            s1 = self.W1(H)
            s2 = self.W2(nn.functional.tanh(s1))
            A = nn.functional.softmax(s2.t())

            #GET EMBEDDING MATRIX GIVEN ATTENTION WEIGHTS
            M = torch.mm(A, H)
            Batched_Attentions[i, :] = M.view(-1)

            # PENALIZATION = FrobNorm((A*A^T) - I)
            AAT = torch.mm(A, A.t())
            P = torch.norm( AAT - I, 2 )
            penal += P * P
            weights[i] = A

        # Penalization Term
        penal /= input.size( 0 )

        # DECODING ATTENTION EMBEDDED MATRICES TO OUTPUT
        MLPhidden = self.MLP(Batched_Attentions)
        decoded = self.normalize(self.decoder(nn.functional.relu(MLPhidden)))

        return decoded, h, penal, weights

