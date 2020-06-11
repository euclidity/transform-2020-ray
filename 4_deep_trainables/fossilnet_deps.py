

from dependencies import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

def get_data_loaders(fossilnet_path,
                     batch_size=16,
                     augment_flip=True,
                     use_grayscale=True):
    
    #
    # We define an array (pipeline) of transformers that we then use Compose to present to the dataset
    #
    txs = []
    
    if use_grayscale:
        # convert to gray but maintain 3 channels for resnet
        txs.append(transforms.Grayscale(3))
    
    if augment_flip:
        txs.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
    
    txs.append(transforms.ToTensor())
    
    if use_grayscale:
        txs.append(transforms.Normalize(0.0, 1.0))

    
    #
    # Use the torchvision ImageFolder Dataset class
    #
    train_dataset = datasets.ImageFolder(
                            root=path.join(fossilnet_path, 'train'),
                            transform=transforms.Compose(txs)
                        )
    
    #
    # Setup a DataLoader to get batches of images for training
    #
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=3,
                              pin_memory=True)
    
    
    #
    # Setup a DataSet and Loader for the test data. This time without shuffle or
    # augmentations enabled
    #
    val_txs = []
    
    if use_grayscale:
        val_txs.extend([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(0.0, 1.0)
        ])
    else:
        val_txs.append(transforms.ToTensor())
    
    val_dataset = datasets.ImageFolder(
                            root=path.join(fossilnet_path, 'test'),
                            transform=transforms.Compose(val_txs)
                        )
    
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True)
    
    return train_loader, val_loader



# cue cool name
class FossilResNet(nn.Module):
    
    def __init__(self, num_outputs=10):
        super(FossilResNet, self).__init__()
        
        # this will pull the weights down to a local cache on first execution
        self.model_conv = torchvision.models.resnet18(pretrained=True)
        
        # we turn of gradients on all layers, so the optimiser will ignore them during backward
        for param in self.model_conv.parameters():
            param.requires_grad = False

        # we replace the last layer with a freshly initialised one, targeting the correct number of outputs,
        # which will be optimised
        num_ftrs = self.model_conv.fc.in_features
        self.model_conv.fc = nn.Linear(num_ftrs, num_outputs)
        
    def forward(self, x):
        return self.model_conv(x)
        
        
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score

def train(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")

    #
    # Accumulate labels and predicitons manually over all batches
    #
    y_all = []
    y_class = []
    
    # iterate over all training batches
    model.train()
    for X, y in tqdm(train_loader, desc="Training..."):

        # send data to the gpu
        X, y = X.to(device), y.to(device)
        
        # zero gradients from last step
        optimizer.zero_grad()
        
        # run forward pass
        y_pred = model(X)
        
        # compute the loss
        loss = F.nll_loss(y_pred, y)
        
        # backpropagate
        loss.backward()
        
        # step the optimiser
        optimizer.step()
        
        # keep hold of target and compute y_class for metrics
        y_all.extend(y.tolist())   
        _, c = torch.max(y_pred, 1)
        y_class.extend(c.tolist())
        
    #
    # Compute f1 on all examples
    #
    return f1_score(y_all, y_class, average='micro')



def validate(model, data_loader, device=None):
    device = device or torch.device("cpu")
    
    #
    # Accumulate labels and predicitons manually over all batches
    #
    y_all = []
    y_class = []
        
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(data_loader, desc="Testing..."):
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X).cpu()
            
            # keep hold of target and compute y_class for metrics
            y_all.extend(y.tolist())   
            _, c = torch.max(y_pred, 1)
            y_class.extend(c.tolist())
            
    #
    # Compute f1 on all examples
    #

    return f1_score(y_all, y_class, average='micro')



from os import path

class FossilTrainable(tune.Trainable):
    
    def _setup(self, config):
        # detect if cuda is availalbe as ray will assign GPUs if available and configured
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_loader, self.test_loader = get_data_loaders(
            #
            # This path needs to be right for you local system
            #
            path.expanduser('~/dev/swung/transform-2020-ray/datasets/fossilnet/tvt_split/4'),
            batch_size=int(config.get("batch_size", 16)),
            augment_flip=config.get("augment_flip", True),
            use_grayscale=config.get("use_grayscale", True)
        )
        
        #
        # Create the network
        #
        self.model = FossilResNet(num_outputs=4).to(self.device)
        
        #
        # Setup the optimiser
        #
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            weight_decay=config.get("weight_decay", 1e-5)
        )

        #
        # Use Trainable state to keep track of best scores
        #
        self._best_train_f1_score = 0.
        self._best_val_f1_score = 0.
        
    def _train(self):
        train_f1_score = train(self.model,
                               self.optimizer,
                               self.train_loader,
                               device=self.device)
        
        val_f1_score = validate(self.model,
                                self.test_loader,
                                self.device)
        
        if (train_f1_score > self._best_train_f1_score):
            self._best_train_f1_score = train_f1_score
        
        if (val_f1_score > self._best_val_f1_score):
            self._best_val_f1_score = val_f1_score
        
        #
        # Really we should return losses here too and we
        # are free to extend the return dict with anything we want to track
        #
        return dict(
            train_f1_score=train_f1_score,
            best_train_f1_score = self._best_train_f1_score,
            val_f1_score=val_f1_score,
            best_val_f1_score=self._best_val_f1_score
        )

    def _save(self, checkpoint_dir):
        checkpoint_path = path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path
    
    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
        
        
