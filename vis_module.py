import torch 
import torch.nn as nn
import torch.nn.functional as F

class vision_module(nn.Module):
    def __init__(self, batch_size, num_classes):
        super(vision_module, self).__init__()
        # init all layers
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)

        self.dense1 = nn.Linear(in_features = 32*29*29, out_features = 64)
        self.dense2 = nn.Linear(in_features = 64, out_features = 16)

        self.classification = nn.Linear(in_features = 16, out_features = num_classes)
        #self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        x = x.float()
        #print("initial shape: ", x.shape)
        out = F.relu(self.conv1(x))
        #print("after conv1 shape: ", out.shape)
        out = self.pool(out)
        #print("after pool1 shape: ", out.shape)
        out = F.relu(self.conv2(out))
        #print("after conv2 shape: ", out.shape)

        out = out.view(-1, 32*29*29)
        out = F.relu(self.dense1(out))
        #print("after dense1 shape: ", out.shape)
        out = F.relu(self.dense2(out))
        #print("after dense2: ", out.shape)

        out = self.classification(out)
        #print("after classification shape: ", out.shape)
        #out = self.softmax(out)

        return out