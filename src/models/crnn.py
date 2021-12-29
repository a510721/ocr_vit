#https://github.com/Deepayan137/Adapting-OCR/tree/master/src

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import random

from argparse import ArgumentParser
from src.options.opts import base_opts
from src.models.lstm import BidirectionalLSTM

class CRNN(nn.Module):

    def __init__(self, opt, leakyRelu=False):
        super(CRNN, self).__init__()

        assert opt.imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = opt.nChannels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64

        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32

        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16

        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16

        convRelu(6, True)  # 512x1x16


        self.cnn = cnn
        self.rnn = nn.Sequential()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(opt.nHidden*2, opt.nHidden, opt.nHidden),
            BidirectionalLSTM(opt.nHidden,   opt.nHidden, opt.nClasses)
        )

    def forward(self, input):
        #conv features
        features = self.cnn(input)
        b, c, h ,w = features.size()
        assert h == 1, "the height of conv must be 1"
        features = features.squeeze(2)
        features = features.permute(2,0,1) # [w, b, c] 33(w:length), 1(batch), 512(c)

        # (L,N,Hin) L:sequence length, N = batch size, Hin = input size(feature)

        # rnn features
        output = self.rnn(features)
        output = output.transpose(1,0) #Tbh to bth(batch, length, ouput) [1(b),33(w),28(class)]

        return output

if __name__ == '__main__':


    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    args.nClasses = 81

    input = torch.rand(1,1,32,128)
    model = CRNN(args)
    output = model(input)
    print('------------------')


'''
if __name__ == '__main__':
   
    # N = batch size, L = sequence length, Hin = input size
    # input_size – The number of expected features in the input x
    # hidden_size – The number of features in the hidden state h
    # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM
    # D =2 if bidirectional = True,otherwise 1
    # Hout = proj_size if proj_size>0 otherwise hidden size
    rnn = nn.LSTM(10, 20, 2) # input_size, hidden_size, num_layers
    input = torch.randn(5, 3, 10) #(L,N,Hin) batch_first=False
    h0 = torch.randn(2, 3, 20) #D∗num_layers,N,Hout
    c0 = torch.randn(2, 3, 20) #D∗num_layers,N,Hcell
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output.size()) #L,N,D*Hout
'''
