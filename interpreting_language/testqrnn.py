import torch
from torchqrnn import QRNN

seq_len, batch_size, hidden_size = 7, 20, 256
size = (seq_len, batch_size, hidden_size)
X = torch.autograd.Variable(torch.rand(size), requires_grad=True).cuda()

qrnn = QRNN(hidden_size, hidden_size, num_layers=2, dropout=0.4)
qrnn.cuda()
output, hidden = qrnn(X)

print(output.size(), hidden.size())
