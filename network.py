import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


class YOLO_V1(nn.Module):
    def __init__(self):
        super(YOLO_V1, self).__init__()
        C = 24  # number of classes
        print("\n------Initiating YOLO v1------\n")
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
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.flatten = Flatten()
        self.conn_layer1 = nn.Sequential(
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1)
        )
        self.conn_layer2 = nn.Sequential(nn.Linear(in_features=4096, out_features=7 * 7 * (2 * 5 + C)))

    def forward(self, input):
        #print(input.size())
        conv_layer1 = self.conv_layer1(input)
        #print(conv_layer1.size())
        conv_layer2 = self.conv_layer2(conv_layer1)
        #print(conv_layer2.size())
        conv_layer3 = self.conv_layer3(conv_layer2)
        #print(conv_layer3.size())
        conv_layer4 = self.conv_layer4(conv_layer3)
        #print(conv_layer4.size())
        conv_layer5 = self.conv_layer5(conv_layer4)
        #print(conv_layer5.size())
        conv_layer6 = self.conv_layer6(conv_layer5)
        #conv_layer6 = self.conv_layer6(conv_layer4)
        #print(conv_layer6.size())
        flatten = self.flatten(conv_layer6)
        #flatten = self.flatten(conv_layer4)
        #print(flatten.size())
        conn_layer1 = self.conn_layer1(flatten)
        output = self.conn_layer2(conn_layer1)
        return output



    def load_conv_bn(self, buf, start, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()
        bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
        bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
        bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
        bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
        #conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
        conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data)); start = start + num_w
        return start

    def load_conv(self, buf, start, conv_model):

        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        #print("start: {}, num_w: {}, num_b: {}".format(start, num_w, num_b))
        # by ysyun, use .view_as()
        conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(conv_model.bias.data));   start = start + num_b
        conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight.data)); start = start + num_w
        return start

    def load_fc(self, buf, start, fc_model):
        num_w = fc_model.weight.numel()
        num_b = fc_model.bias.numel()
        fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
        fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]));   start = start + num_w 
        return start

    def load_weight(self,weight_file):
        print("Load pretrained models !")

        fp = open(weight_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        header = torch.from_numpy(header)
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()
        a = self.children()

        start = 0
        for idx,q in enumerate(a):
            print(q)
            for m in q.children():
                if isinstance(m, nn.Conv2d):
                    conv = m
                    start = self.load_conv(buf, start, conv)
                elif isinstance(m, nn.BatchNorm2d):
                    start = self.load_conv_bn(buf, start, conv, m)
                # elif isinstance(m, nn.Linear):
                #     start = self.load_fc(buf, start, m)




if __name__ == '__main__':
    net = YOLO_V1()
    #Download weights here: http://pjreddie.com/media/files/yolov1/yolov1.weights
    net.load_weight ('../yolov1.weights')
