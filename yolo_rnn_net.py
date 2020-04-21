import torch.nn as nn
import torch
from torch.autograd import Variable
import tcnnlayer


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
    def forward(self, x):
        batch, seqLen, channels, features1, features2 = x.size()
        return x.view(batch*seqLen, channels, features1, features2), seqLen

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        shape = x.size()
        return x.view(shape[0], -1)

class Flatten2(nn.Module):
    def __init__(self):
        super(Flatten2, self).__init__()
    def forward(self, x):
        shape = x.size()
        return x.contiguous().view(shape[0]*shape[1], -1)

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
    def forward(self, x, seqLen):
        batch, features = x.size()
        return x.view(int(batch/seqLen), seqLen, features)


class rnnNorm(nn.Module):
    def __init__(self):
        super(rnnNorm, self).__init__()
        self.bn = nn.BatchNorm1d(4096)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(x).detach()
        x = x.permute(0, 2, 1)
        return x

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        else:
            x = x.permute(1,0,2)
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
            mask = Variable(m, requires_grad=False) / (1 - dropout)
            mask = mask.expand_as(x)
            x = mask * x
            del mask
            del m
            return x.permute(1,0,2)


class YOLO_V1(nn.Module):
    def __init__(self, rnn_type="RNN"):
        super(YOLO_V1, self).__init__()
        C = 24  # number of classes
        print("\n------Initiating YOLO v1 with",rnn_type,"layers------\n")
        self.rnn_type = rnn_type
        self.combine = Combine()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=7//2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            #nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=1, stride=1, padding=1//2), #added
            #nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2),
            #nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            #nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            #nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            #nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            #nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            #nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            #nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.conv_layer6 = nn.Sequential(
            #nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            #nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=3//2), # added
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2), # added
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.flatten = Flatten()
        self.conn_layer1 = nn.Sequential(
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1)
        )
        self.squeeze = Squeeze()
        if(rnn_type=="RNN"):
            self.rnn = nn.RNN(input_size=4096 , hidden_size=4096 , num_layers= 1, batch_first=True, nonlinearity="relu")
            self.locked_dropout = LockedDropout()
           # self.rnnNorm = rnnNorm()
        elif(rnn_type=="LSTM"):
            self.conn_shrink = nn.Sequential(
                nn.Linear(in_features=4096, out_features=2048),
                nn.Dropout(),
                nn.LeakyReLU(0.1)
            )
            self.rnn = nn.LSTM(input_size=2048 , hidden_size=4096 , num_layers= 1, batch_first=True)
        elif(rnn_type=="TCNN"):
            self.rnn1 = tcnnlayer.TCNN(4096, 512, 2, p_in=True)
            self.rnn2 = tcnnlayer.TCNN(512, 4096, 2, p_out=True)
        self.flatten2 = Flatten2()
        self.conn_layer2 = nn.Sequential(nn.Linear(in_features=4096, out_features=7 * 7 * (2 * 5 + C)))
        self.squeeze2 = Squeeze()

    def forward(self, input, h_prev=None, same_shape=False):

        newInput, seqLen = self.combine(input)

        conv_layer1 = self.conv_layer1(newInput)

        conv_layer2 = self.conv_layer2(conv_layer1)

        conv_layer3 = self.conv_layer3(conv_layer2)

        conv_layer4 = self.conv_layer4(conv_layer3)

        conv_layer5 = self.conv_layer5(conv_layer4)

        conv_layer6 = self.conv_layer6(conv_layer5)

        flatten = self.flatten(conv_layer6)

        conn_layer1 = self.conn_layer1(flatten)

        if(self.rnn_type=="RNN" or self.rnn_type=="TCNN"):
            squeeze = self.squeeze(conn_layer1, seqLen)
        else:
            conn_shrink = self.conn_shrink(conn_layer1)
            squeeze = self.squeeze(conn_shrink, seqLen)

        if(self.rnn_type=="TCNN"):
            r_1, h_1 = self.rnn1(squeeze, h_prev[0] if h_prev!=None else None)
            r_2, h_2 = self.rnn2(r_1, h_prev[1] if h_prev!=None else None)
            r_out = squeeze+r_2 # residual
            h_n = (h_1, h_2)
        elif(h_prev is None):
            r_out, h_n = self.rnn(squeeze)
        else:
            r_out, h_n = self.rnn(squeeze, h_prev)

        if(self.rnn_type=="RNN"):
           # dropped_r_out = self.locked_dropout(r_out)
           # normed_r_out = self.rnnNorm(dropped_r_out)
            flatten2 = self.flatten2(r_out)
        else:
            flatten2 = self.flatten2(r_out)

        output = self.conn_layer2(flatten2)

        if(same_shape):
            return self.squeeze2(output, seqLen), h_n
        else:
            return output, h_n

