#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: Jitesh G
# DATE CREATED: 03/03/2020                            
# REVISED DATE: 
# PURPOSE: Function loads the model for the neural network to be used for prediction
#          from the checkpoint provided
import torch
from torchvision import models

def load_model(filepath):
    """
        Loads a CNN network from the checkpoint provided. Checkpoint provides the following information:
        1. Network Architecture
        2. Classifier
        
        
        Parameters:
         checkpoint - Contains the relevant items required to recreate the network
         
        Returns:
         model - the loaded model
    """
    
    checkpoint = torch.load(filepath)
    
    if(checkpoint['arch'] == "vgg"):
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model