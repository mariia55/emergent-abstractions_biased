import argparse
import torch
import numpy as np
from tqdm import tqdm
from load import load_data
from vision_module import vision_module
from create_datasets import shapes_dataset

def get_params():

    parser = argparse.ArgumentParser(prog="Train Visual Model", description="trains a vision module")

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the training")
    
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help="Weight decay for the training")
    
    parser.add_argument('--epochs', type=int, default=10, 
                        help="How many epochs to train for")
    
    parser.add_argument('--test_every', type=int, default=5, 
                        help="After how many training epochs each test runs are to be conducted")
    
    parser.add_argument('--save_model', type=bool, default=False,
                        help="Use if you want to save the model after training")
    
    parser.add_argument('--train_split', type=float, default=0.8,
                        help="Determine how much of the dataset will be used for training and how much for testing")
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Give batch size for training")
    
    parser.add_argument('--generate_dataset', type=bool, default=False,
                        help="Determine whether to generate the dataset with the saved model")
    
    args = parser.parse_args()

    return args


def train(args):
    # first load the dataset and define all parts necessary for the training
    # try to load the dataset if it was saved before
    try:

        complete_data = torch.load('./dataset/complete_dataset')

        print('Dataset was found and loaded successfully')

    # otherwise create the dataset and save it to the folder for repeated use
    except:

        print('Dataset was not found, creating it instead...')
        input_shape = [3,64,64]
        
        full_data, labels_reg, full_labels = load_data(input_shape, normalize=False,
                                                                        subtract_mean=False,
                                                                        trait_weights=None,
                                                                        return_trait_weights=False,
                                                                        return_full_labels=True,
                                                                        datapath=None)
        
        complete_data = shapes_dataset(full_data, labels_reg)

        torch.save(complete_data, './dataset/complete_dataset')

        print('Dataset was created and saved in /dataset')


    train_size = int(args.train_split * len(complete_data))
    test_size = len(complete_data) - train_size
    
    train_data, test_data = torch.utils.data.random_split(complete_data, [train_size, test_size])

    # init both dataloaders with corresponding datasets
    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size = args.batch_size, 
                                            shuffle = True,
                                            pin_memory = True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size = args.batch_size, 
                                            shuffle = False,
                                            pin_memory = True)

    # init vision module to train
    model = vision_module(num_classes=64)

    # use gpu if possible
    if torch.cuda.is_available():
            model.cuda()

    # use SGD as optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr = args.learning_rate,
                                weight_decay = args.weight_decay)
    
    # use crossentropyloss as loss
    criterion = torch.nn.CrossEntropyLoss()

    # now after everything was set up the actual training starts
    # save losses and accuracies as lists for later printing
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # train for given nr of epochs
    print('Starting training for ', args.epochs, ' epochs\n')
    for epoch in range(args.epochs):
        print("Epoch ", epoch+1)

        # lists to calculate the values for each
        temp_losses = []
        temp_accs = []

        model.train()
        
        for i, (input, target) in tqdm(enumerate(train_loader)):

            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            
            # compute network prediction and use it for loss
            output = model(input)
            loss = criterion(output, target)
            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute accuracy for each single entry per batch
            for single_output, single_target in zip(output, target):
                if np.argmax(single_output.cpu().detach().numpy()) == np.argmax(single_target.cpu().detach().numpy()):
                    temp_accs.append(1.0)
                else:
                    temp_accs.append(0.0)
            
            temp_losses.append(float(loss))

        # add accuracy and loss for each epoch
        train_accuracies.append(sum(temp_accs)/len(temp_accs))
        train_losses.append(sum(temp_losses)/len(temp_losses))

        # print most recent entry to our loss and accuracy lists
        print("Epoch ", epoch+1, " achieved training loss: ~", round(train_losses[-1], 3)," and training accuracy: ~", round(train_accuracies[-1], 3), "\n")

        # test every X epoch as given by test_every
        if ((epoch+1) % args.test_every) == 0:

            test_loss, test_acc = test(test_loader, model, criterion)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
    
    if args.save_model:
        model.cpu()
        torch.save(model.state_dict(), './models/vision_module')
    
    return train_losses, train_accuracies, test_losses, test_accuracies

def test(test_loader, model, criterion):    

    model.eval()

    with torch.no_grad():
         print('Starting test run')
         temp_losses = []
         temp_accs = []
         for i, (input, target) in tqdm(enumerate(test_loader)):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            
            # compute network prediction and use it for loss
            output = model(input)
            loss = criterion(output, target)

            # compute accuracy for each single entry per batch
            for single_output, single_target in zip(output, target):
                if np.argmax(single_output.cpu().detach().numpy()) == np.argmax(single_target.cpu().detach().numpy()):
                    temp_accs.append(1.0)
                else:
                    temp_accs.append(0.0)

            temp_losses.append(float(loss))

    # add accuracy and loss
    loss = sum(temp_losses)/len(temp_losses)
    acc = sum(temp_accs)/len(temp_accs)

    print("test loss was: ~", round(loss, 3), "and test accuracy was: ~", round(acc, 3), "\n")
    
    return loss, acc

if __name__ == "__main__":

    args = get_params()

    train_losses, train_accuracies, test_losses, test_accuracies = train(args)

    # round losses and accuracies for prettier printing
    rounded_train_losses        = [round(loss, 3) for loss in train_losses]
    rounded_train_accuracies    = [round(acc, 3) for acc in train_accuracies]
    rounded_test_losses         = [round(loss, 3) for loss in test_losses]
    rounded_test_accuracies     = [round(acc, 3) for acc in test_accuracies]

    print('training losses were: ~', rounded_train_losses, "\n")
    print('training accuracies were: ~', rounded_train_accuracies, "\n")

    print('test losses were: ~', rounded_test_losses, "\n")
    print('test accuracies were: ~', rounded_test_accuracies, "\n")