from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
import img_dataset
import torch
import torchmetrics

data = img_dataset.shapes3d(path_to_dataset = '3dshapes/3dshapes.h5', transform = None)
train_loader = torch.utils.data.DataLoader(data,
                                           batch_size = 32, 
                                           shuffle = True,
                                           pin_memory = True)
test_loader = torch.utils.data.DataLoader(data,
                                          batch_size = 32, 
                                          shuffle = False,
                                          pin_memory = True)

network = resnet20()

criterion = torch.nn.CrossEntropyLoss()

lr = 0.001


optimizer = torch.optim.Adam(network.parameter(),
                             lr = 0.001,
                             weight_decay = 0.0001)

def train(train_loader, network, criterion, optimizer, epochs):
    network.train()
    
    for epoch in range(epochs):
        losses = []
        accuracies = []
        for i, (input, target) in enumerate(train_loader):
            output = network(input)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=6)
            accuracy(output, target)
            accuracies.append(accuracy)
