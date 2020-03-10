#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: Jitesh G
# DATE CREATED: 02/03/2020                            
# REVISED DATE: 
# PURPOSE: Function creates the model for the neural network. With the objective
#          to implement transfer learning, a CNN is downloaded from the models library.
#          A classifier is created and attached on to the pre-trained network.


from torchvision import models
from collections import OrderedDict
from torch import nn
from torch import optim



def create_model(arch, hidden_units):
    """
        Creates a CNN network by leveraging a pretrained network and attaching a custom classifier.
        The pre-trained networks can be:
        1. vgg16
        2. densenet121

        Parameters:
         arch - specifies the pre-trained model to be used (vgg16/densenet121 only)
         hidden_units - number of features the hidden layer should have
        Returns:
         model - the newly created model
    """

    # import model from torchvision
    if(arch == "vgg"):
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[0].in_features
    else:
        model = models.densenet121(pretrained=True)
        num_features = model.classifier.in_features

    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # create classifier    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout',nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # attach classifier to pre-trained vgg16 network
    model.classifier = classifier

    


    return model