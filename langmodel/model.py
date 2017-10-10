import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LangModel(nn.Module):
    
    def __init__(
            self,
            vocab_size,
            model_type = 'LSTM', 
            input_size = 300, 
            hidden_size = 4096,
            num_layers = 1,
            rnn_dropout = 0.5,
            linear_dropout = 0.5,
            tie_weights = False,
            init_range = 0.1
        ):

        super(LanguageModel, self).__init__()
        
        self.model = getattr(nn, model_type)(input_size, hidden_size, num_layers, dropout = rnn_dropout)
        
        self.drop = linear_dropout
        self.linear = nn.Linear(hidden_size, vocab_size)

        if tie_weights:
            assert hidden_size == input_size
            self.decoder.weight = self.encoder.weight

        self.init_weights(init_range)
        
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self, init_range):
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange, initrange)

    def init_hidden(batch_size):
        num_states = 1
        if self.model_type == 'LSTM':
            num_states = 2
        
        return (Variable(torch.zeros(1, 1, self.hidden_size)) for i in range(num_states))


    def forward(self, inp, h):
        out, h = self.model(inp, h)
        out = self.drop(out)
        
        predictions = self.linear(out.view(out.size(0)*out.size(1), out.size(2)))
        
        return predictions.view(out.size(0), out.size(1), out.size(2))
