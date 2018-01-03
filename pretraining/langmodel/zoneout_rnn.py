import torch.nn as nn
import torch

#ZONEOUT LSTM Implementation from https://discuss.pytorch.org/t/implementing-recurrent-dropout/5343/2

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                 norms=cfg.lstm.norms,
                 tie_forget=cfg.lstm.tie_forget,
                 forget_bias=cfg.lstm.forget_bias,
                 activation_function=cfg.lstm.activation_function):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.norms = norms
        self.tie_forget = tie_forget
        self.forget_bias = forget_bias
        self.af = activation_function

        self.matrix_width = 3 if self.tie_forget else 4

        self.combined_weights = nn.Parameter(
            torch.FloatTensor(self.input_size + self.hidden_size, self.matrix_width * self.hidden_size))

        if 'batch' in self.norms:
            self.bn = nn.BatchNorm1d(self.matrix_width * self.hidden_size)
            self.bn_c = nn.BatchNorm1d(self.hidden_size)
        else:
            self.bias = nn.Parameter(torch.FloatTensor(self.matrix_width * self.hidden_size))

        # This seems like a hacky way to implement zoneout, but I'm not sure what the correct way would be
        self.register_buffer('bernoulli_mask',
                             torch.Tensor(1).fill_(cfg.lstm.zoneout).expand((self.hidden_size,)))

        self.reset_parameters()

    def reset_parameters(self):

        # Initialize combine_weights
        weight_ih_data = init.orthogonal(torch.Tensor(self.input_size, self.matrix_width * self.hidden_size))
        weight_hh_data = torch.eye(self.hidden_size).repeat(1, self.matrix_width)
        self.combined_weights.data.set_(torch.cat((weight_hh_data, weight_ih_data), 0))

        if 'batch' in self.norms:
            self.bn.reset_parameters()
            self.bn_c.reset_parameters()
            self.bn.bias.data[0:self.hidden_size].fill_(self.forget_bias)
        else:
            self.bias.data.fill_(0)
            self.bias.data[0:self.hidden_size].fill_(self.forget_bias)


    def forward(self, input, hx):
        """
        Args:
            input: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        combined_inputs = torch.cat((h_0, input), 1)

        if 'batch' in self.norms:
            preactivations = torch.mm(combined_inputs, self.combined_weights)
            preactivations = self.bn(preactivations)
        else:
            batch_size = h_0.size(0)
            bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
            preactivations = torch.addmm(bias_batch, combined_inputs, self.combined_weights)

        if self.tie_forget:
            fi, o, g   = torch.split(preactivations, split_size=self.hidden_size, dim=1)
            c_1 = torch.sigmoid(fi)*c_0 + torch.sigmoid(1-fi)*self.af(g)
        else:
            f, i, o, g = torch.split(preactivations, split_size=self.hidden_size, dim=1)
            c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*self.af(g)

        h_1 = torch.sigmoid(o) * self.af(self.bn_c(c_1) if 'batch' in self.norms else c_1)

        if cfg.lstm.zoneout > 0:
            if cfg.training:
                h_mask = Variable(torch.bernoulli(self.bernoulli_mask))
                c_mask = Variable(torch.bernoulli(self.bernoulli_mask))
                h_1 = h_0 * h_mask + h_1 * (1-h_mask)
                c_1 = c_0 * c_mask + c_1 * (1-c_mask)
            else:
                h_1 = h_0 * cfg.lstm.zoneout + h_1 * (1-cfg.lstm.zoneout)
                c_1 = c_0 * cfg.lstm.zoneout + c_1 * (1-cfg.lstm.zoneout)

        return h_1, c_1
