import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from load_frames import *
from YoloLoss import YoloLoss
from mini import *



loss_name = 'loss_mini.h5'
model_name = 'model_mini.pth'

 ### time start
start_time = time.time()


# ### gpu usage
use_gpu = torch.cuda.is_available()


# ### dataset and file folder
annotDir = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/train/annots/"
videoDir = "/media/trocket/27276136-d5a4-4943-825f-7416775dc262/home/trocket/data/train/videos/"


# ### sample dataset
# annotDir = "sample_data/train/annots/"
# videoDir = "sample_data/train/videos/"


# ### set hyperparameters
learning_rate = 0.0006
img_size = 224
num_epochs = 150
lambda_coord = 5
lambda_noobj = .5
#n_batch = 64
n_batch = 32
S = 7 # This is currently hardcoded into the YOLO model
B = 2 # This is currently hardcoded into the YOLO model
C = 24 # This is currently hardcoded into the YOLO model
n_features = 1000



# load mini yolo model
model = YOLO_V1()
print(model)
print("untrained YOLO_V1 model has loaded! (mini version)")
print("")



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    model = nn.DataParallel(model).cuda()
#else:
#    model.to(device)
model.to(device)


# ### input pipeline
train_dataset = FramesDataset(videoDir=videoDir, annotDir=annotDir, img_size=img_size, S=S, B=B, C=C, transforms=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True, num_workers=4)



# ### set model into train mode
model.train()


# ### set loss function and optimizer
loss_fn = YoloLoss(n_batch, B, C, lambda_coord, lambda_noobj, use_gpu=use_gpu)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)

save_folder = 'results/'

# ### training
loss_list = []
loss_record = []
for epoch in range(num_epochs):
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.to(device),target.to(device)

        pred = model(images)
        loss = loss_fn(pred,target)

        current_loss = loss.data.cpu().numpy()
        loss_list.append(current_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % 10 == 0:
            sys.stdout.write("\r%d/%d batches in %d/%d iteration, current error is %f"                              % (i, len(train_loader), epoch+1, num_epochs, current_loss))
            sys.stdout.flush()
        loss_record.append(current_loss)
        torch.save(model.state_dict(),os.path.join(save_folder, model_name))


# ### save the model parameters
save_folder = 'results/'

loss_list = np.array(loss_list)
dd.io.save(os.path.join(save_folder, loss_name), loss_list)

print('loss has saved successfully!')


# ### save the model parameters
# set model into eval mode
model.eval()

torch.save(model.state_dict(),os.path.join(save_folder, model_name))

loss_record = np.array(loss_record)
dd.io.save(os.path.join(save_folder, 'mini_loss_150epoches_0411.h5'), loss_record)

print('model has saved successfully!')


# ### time end
print("\n--- it costs %.4s minutes ---" % ((time.time() - start_time)/60))

