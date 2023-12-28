from vis_module import vision_module
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

network = vision_module(batch_size=32, num_classes=6)

criterion = torch.nn.CrossEntropyLoss()

lr = 0.001


optimizer = torch.optim.Adam(network.parameters(),
                             lr = 0.001,
                             weight_decay = 0.0001)

def train(train_loader, network, criterion, optimizer, epochs):
    print("why did we start?!")
    losses = []
    accuracies = []
    network.train()
    
    for epoch in range(epochs):
        print("starting epoch ", epoch+1)
        
        for i, (input, target) in enumerate(train_loader):
            output = network(input)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=6)
            accuracy(output, target)
            accuracies.append(accuracy)
            losses.append(loss)

train(train_loader, network, criterion, optimizer, epochs = 5)