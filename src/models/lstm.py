import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, nIn, nHidden):
        super(SimpleLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)

    def forward(self, input_):
        recurrent, _ = self.rnn(input_)
        T,b,h = recurrent.size()
        return recurrent

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM,self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden*2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T,b,h = recurrent.size()
        t_rec = recurrent.view(T*b,h)
        output = self.embedding(t_rec) #[T*b, nOut)
        output = output.view(T,b,-1)#[T,b,nOut]
        return output
