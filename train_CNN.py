from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
import dataset

train_data = dataset.DataSet
train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size = args.batch_size, 
                                               shuffle = True,
                                               num_workers = args.workers, 
                                               pin_memory = True)