import torch 
import torch.nn as nn
import torch.nn.functional as F

class vision_module(nn.Module):
    def __init__(self, batch_size, num_classes):
        super(vision_module, self).__init__()
        # init all layers
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.dense1 = nn.Linear(in_features = 32*29*29, out_features = 180)
        self.dense2 = nn.Linear(in_features = 180, out_features = 100)

        self.classification = nn.Linear(in_features = 100, out_features = num_classes)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        x = x.float()
        out = self.pool(F.relu(self.conv1(x)))
        out = F.relu(self.conv2(out))

        # flatten input data here
        out = out.view(-1, 32*29*29)
        # and use flattened data for following classification layers
        out = F.relu(self.dense1(out))
        out = F.relu(self.dense2(out))

        out = self.classification(out)
        out = self.softmax(out)

        return out