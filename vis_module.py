import torch 
import torch.nn as nn
import torch.nn.functional as F

class vision_module(nn.Module):
    def __init__(self, batch_size, num_classes):
        super(vision_module, self).__init__()
        # init all layers
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)

        self.dense1 = nn.Linear(in_features = 32, out_features = 16)
        self.dense2 = nn.Linear(in_features = 16, out_features = 16)

        self.classification = nn.Linear(in_features = 16, out_features = num_classes)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        #print("this is x: ", x)
        #print("this is it's type : ", type(x))
        x = x.float()
        print("this is it's shape: ", x.shape)
        out = F.relu(self.conv1(x))
        print("that fixed it?!")
        out = F.relu(self.conv2(out))

        out = F.relu(self.dense1(out))
        out = F.relu(self.dense1(out))

        out = self.classification(out)
        out = self.softmax(out)

        return out