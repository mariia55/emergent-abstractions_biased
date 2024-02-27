import argparse
import torch
import numpy as np
from tqdm import tqdm
from load import load_data
from vis_module import vision_module
from img_dataset import shapes3d

def get_params():

    parser = argparse.ArgumentParser(prog="Train Visual Model", description="trains a vision module")

    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the training")
    
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help="Weight decay for the training")
    
    # parser.add_argument('--test_size', type=float, default=0.25,
    #                     help="Part of the full dataset to be used as the test part when splitting into training and test_set")
    
    parser.add_argument('--epochs', type=int, default=10, 
                        help="How many epochs to train for")
    
    parser.add_argument('--test_every', type=int, default=5, 
                        help="After how many training epochs each test runs are to be conducted")
    
    parser.add_argument('--save_model', type=bool, default=False,
                        help="Use if you want to save the model after training")
    
    args = parser.parse_args()

    return args


def train(args):
    # first load the dataset and define all parts necessary for the training
    # try to load the dataset if it was saved before
    try:
        train_data = torch.load('./dataset/training_dataset')
        validation_data = torch.load('./dataset/validation_dataset')
        test_data = torch.load('./dataset/test_dataset')

        print('Dataset was found and loaded successfully')

    # otherwise create the dataset and save it to the foulder for use in later runs
    except:
        print('Dataset not found, creating it instead...')
        input_shape = [3,64,64]
        
        train_data, validation_data, test_data, target_names, full_labels, complete_data = load_data(input_shape, normalize=False,
                                                                        subtract_mean=False,
                                                                        trait_weights=None,
                                                                        return_trait_weights=False,
                                                                        return_full_labels=True,
                                                                        datapath=None)
        
        torch.save(train_data, './dataset/training_dataset')
        torch.save(validation_data, './dataset/validation_dataset')
        torch.save(test_data, './dataset/test_dataset')
        torch.save(complete_data, './dataset/complete_dataset')
    
    # if (len(train_data)+len(validation_data))*args.test_size != len(validation_data):
    #     print("Found dataset's split did not match given test size of ", args.test_size, ", creating new dataset accordingly...")
    #     input_shape = [3,64,64]

    #     train_data, validation_data, test_data, target_names, full_labels, complete_data = load_data(input_shape, normalize=False,
    #                                                                 subtract_mean=False,
    #                                                                 trait_weights=None,
    #                                                                 return_trait_weights=False,
    #                                                                 return_full_labels=True,
    #                                                                 datapath=None)
    
    #     torch.save(train_data, './dataset/training_dataset')
    #     torch.save(validation_data, './dataset/validation_dataset')
    #     torch.save(test_data, './dataset/test_dataset')


    # init both dataloaders with corresponding datasets
    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size = 32, 
                                            shuffle = True,
                                            pin_memory = True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size = 32, 
                                            shuffle = False,
                                            pin_memory = True)
    
    val_loader = torch.utils.data.DataLoader(validation_data,
                                            batch_size = 32, 
                                            shuffle = False,
                                            pin_memory = True)

    # init vision module to train
    model = vision_module(batch_size=32, num_classes=64)

    # use gpu whenever possible
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

            # compute accuracy
            if np.argmax(output.cpu().detach().numpy()[1]) == np.argmax(target.cpu().detach().numpy()[1]):
                temp_accs.append(1.0)
            else:
                temp_accs.append(0.0)
            
            temp_losses.append(float(loss))
            

        # add accuracy and loss for each epoch
        train_accuracies.append(sum(temp_accs)/len(temp_accs))
        train_losses.append(sum(temp_losses)/len(temp_losses))

        # print most recent entry to our loss and accuracy lists
        print("Epoch ", epoch+1, " achieved training loss: ", round(train_losses[-1], 4)," and training accuracy: ", round(train_accuracies[-1], 4), "\n")

        # test every X epoch as given by test_every
        if ((epoch+1) % args.test_every) == 0:
            test_loss, test_acc = test(test_loader, model, criterion)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
    
    if args.save_model:
        torch.save(model.state_dict(), './models/vision_module')
    
    return train_losses, train_accuracies, test_losses, test_accuracies

def generate_dataset(save = True):
    model = vision_module(batch_size=32, num_classes=64)
    model.load_state_dict(torch.load('./models/vision_module', map_location=torch.device('cpu')))

    if torch.cuda.is_available():
            model.cuda()
    
    model.eval()
    try:
        data = torch.load('./dataset/complete_dataset')
    except:
        print('Dataset not found, creating it instead...')
        input_shape = [3,64,64]
        
        train_data, validation_data, test_data, target_names, full_labels, complete_data = load_data(input_shape, normalize=False,
                                                                        subtract_mean=False,
                                                                        trait_weights=None,
                                                                        return_trait_weights=False,
                                                                        return_full_labels=True,
                                                                        datapath=None)
        
        torch.save(train_data, './dataset/training_dataset')
        torch.save(validation_data, './dataset/validation_dataset')
        torch.save(test_data, './dataset/test_dataset')
        torch.save(complete_data, './dataset/complete_dataset')
        data = complete_data

    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size = 32, 
                                            shuffle = False,
                                            pin_memory = True)
    
    inputs = []
    generated_labels = []
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            print("starting batch nr ", i+1)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            
            output = model(input)

            input_flat = torch.flatten(input, start_dim=0, end_dim=0)
            output_flat = torch.flatten(output, start_dim=0, end_dim=0)
            target_flat = torch.flatten(target, start_dim=0, end_dim=0)

            # print("input flattened: ", input_flat)
            # print("input flat len: ", len(input_flat))
            # print("output flattened: ", output_flat)
            # print("output flat len: ", len(output_flat))
            # print("target flattened: ", target_flat)
            # print("target flat len: ", len(target_flat))

            # this is where input and generated labels "become" tensors
            for input in input_flat:
                inputs.append(input.numpy())
            for output in output_flat:
                generated_labels.append(output.numpy())
    if save:
        # generated labels are tensors while original labels were numpy arrays
        # could lead to problems?
        generated_dataset_full = shapes3d(inputs, generated_labels)
        torch.save(generated_dataset_full, './dataset/generated_dataset')



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

            # compute accuracy
            # is accuracy wrong????
            if np.argmax(output.cpu().detach().numpy()[1]) == np.argmax(target.cpu().detach().numpy()[1]):
                temp_accs.append(1.0)
            else:
                temp_accs.append(0.0)

            temp_losses.append(float(loss))


    # add accuracy and loss
    loss = sum(temp_losses)/len(temp_losses)
    acc = sum(temp_accs)/len(temp_accs)

    print("test loss was: ", round(loss, 4), "and test accuracy was: ", round(acc, 4), "\n")
    
    return loss, acc


if __name__ == "__main__":
    #generate_dataset()
    data = torch.load('./dataset/complete_dataset')
    generated_data = torch.load('./dataset/generated_dataset')

    test = []

    for (original, generated) in zip(data,generated_data):
        #print('original: ', np.argmax(original[1]))
        #print('generated: ', np.argmax(generated[1]))

        if np.argmax(original[1]) == np.argmax(generated[1]):
            test.append(1)
        else:
            test.append(0)

    print(sum(test))
    print(len(test))

    # args = get_params()

    # train_losses, train_accuracies, test_losses, test_accuracies = train(args)

    # # round losses and accuracies for prettier printing
    # rounded_train_losses        = [round(loss, 4) for loss in train_losses]
    # rounded_train_accuracies    = [round(acc, 4) for acc in train_accuracies]
    # rounded_test_losses         = [round(loss, 4) for loss in test_losses]
    # rounded_test_accuracies     = [round(acc, 4) for acc in test_accuracies]

    # print('training losses were: ', rounded_train_losses, "\n")
    # print('training accuracies were: ', rounded_train_accuracies, "\n")

    # print('test losses were: ', rounded_test_losses, "\n")
    # print('test accuracies were: ', rounded_test_accuracies, "\n")