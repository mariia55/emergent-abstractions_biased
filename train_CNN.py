from vis_module import vision_module
import img_dataset
import torch
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy

dataset = img_dataset.shapes3d(path_to_dataset = '3dshapes/3dshapes.h5', transform = None)
# does not work because numpy arrays are too large?
#torch.save(data, './3dshapes/dataset.pt')

# split data into train and test splits
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# init both dataloaders with corresponding datasets
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size = 32, 
                                           shuffle = True,
                                           pin_memory = True)

test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size = 32, 
                                          shuffle = False,
                                          pin_memory = True)

# init vision module
vis_module = vision_module(batch_size=32, num_classes=6)

if torch.cuda.is_available():
        print('using cuda for faster computation')
        vis_module.cuda()

# use crossentropyloss as loss
criterion = torch.nn.CrossEntropyLoss()

# define hyperparameters
lr = 0.01
weight_decay = 0.0001

# use adam as optimizer (maybe try SGD?)
optimizer = torch.optim.SGD(vis_module.parameters(),
                             lr = lr,
                             weight_decay = weight_decay)

# defined training loop
def train(train_loader, network, criterion, optimizer, epochs, test_loader):
    # save losses and accuracies as lists
    losses = []
    accuracies = []
    test_losses = []
    test_accuracies = []

    network.train()
    
    # train for given nr of epochs
    for epoch in range(epochs):
        print("starting epoch ", epoch+1)
        temp_losses = []
        temp_accs = []
        
        for i, (input, target) in tqdm(enumerate(train_loader)):
            if torch.cuda.is_available():
                input.cuda()
                target.cuda()          
            
            # compute network prediction and use it for loss
            output = network(input)
            loss = criterion(output, target)
            temp_losses.append(loss.float())

            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute accuracy for 1 epoch
            accuracy = MulticlassAccuracy(num_classes=10*10*10*8*4*15, average = 'micro', top_k = 1, multidim_average = 'global')
            acc = accuracy(output, target)

            # loss = loss.float()
            # acc = acc.float()
            
            temp_accs.append(acc.float())

        # add accuracy and loss for each epoch
        accuracies.append(sum(temp_accs)/len(temp_accs))
        losses.append(sum(temp_losses)/len(temp_losses))
        print("Epoch ", epoch+1, " had training loss: ", losses[-1], " and training accuracy: ", accuracies[-1])

        if epoch+1 % 2 == 0:
            test_loss, test_acc = test(test_loader, network, criterion)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
    
    return losses, accuracies, test_losses, test_accuracies

def test(test_loader, network, criterion):    
    # save losses and accuracies as lists
    # losses = []
    # accuracies = []

    network.eval()

    with torch.no_grad():
         print('starting test run')
         temp_losses = []
         temp_accs = []
         for i, (input, target) in tqdm(enumerate(test_loader)):
            if torch.cuda.is_available():
                input.cuda()
                target.cuda()
            
            output = network(input)
            loss = criterion(output, target)

            # compute accuracy  10*10*10*8*4*15
            accuracy = MulticlassAccuracy(num_classes=10*10*10*8*4*15, average = 'micro', top_k = 1, multidim_average = 'global')
            acc = accuracy(output, target)

            temp_losses.append(loss.float())
            temp_accs.append(acc.float())

    # add accuracy and loss
    # accuracies.append(sum(temp_accs)/len(temp_accs))
    # losses.append(sum(temp_losses)/len(temp_losses))
    loss = sum(temp_losses)/len(temp_losses)
    acc = sum(temp_accs)/len(temp_accs)
    print("test loss was: ", loss, "and test accuracy was: ", acc)
    
    return loss, acc
              
train_losses, train_accuracies, test_losses, test_accuracies = train(train_loader, vis_module, criterion, optimizer, epochs = 3, test_loader=test_loader)

#test_losses, test_accuracies = test(test_loader, vis_module, criterion)

print('these are the train losses: ', train_losses)
print('these are the train accuracies: ', train_accuracies)

print('these are the test losses: ', test_losses)
print('these are the test accuracies: ', test_accuracies)