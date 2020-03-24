import torch.nn as nn
import torch


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


class YOLO_V1(nn.Module):
    def __init__(self, rnn_type="RNN"):
        super(YOLO_V1, self).__init__()
        C = 24  # number of classes
        print("\n------Initiating YOLO v1 with",rnn_type,"layers------\n")
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
        elif(rnn_type=="LSTM"):
            self.rnn = nn.LSTM(input_size=4096 , hidden_size=4096 , num_layers= 1, batch_first=True)
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

        squeeze = self.squeeze(conn_layer1, seqLen)

        if(h_prev==None):
            r_out, h_n = self.rnn(squeeze)
        else:
            r_out, h_n = self.rnn(squeeze, h_prev)

        flatten2 = self.flatten2(r_out)

        output = self.conn_layer2(flatten2)

        if(same_shape):
            return self.squeeze2(output, seqLen), h_n
        else:
            return output, h_n
