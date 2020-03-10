#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: Jitesh G
# DATE CREATED: 02/03/2020                            
# REVISED DATE: 
# PURPOSE: Function that loads the data from data directory and returns dataloaders.
#          

import torch
from torchvision import datasets, transforms, models

def load_data(data_dir):
    """
    Transforms and loads the data from the directory where flower images are stored.
    Expects data to be split in 3 sub-directories:
    1. train
    2. valid
    3. test

    Parameters:
     data_dir - directory where flower images are stored
     
    Returns:
     trainloader (DataLoader) - train dataset loader
     validloader (DataLoader) - validation dataset loader
     testloader (DataLoader) - test dataset loader
     class_to_idx (dictionary) - mapping of class to indices
    """

        
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # load transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    # image_datasets = 
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    # dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    class_to_idx = train_data.class_to_idx
    return trainloader, validloader, trainloader, class_to_idx
        
        

        
        
        
        

    
   