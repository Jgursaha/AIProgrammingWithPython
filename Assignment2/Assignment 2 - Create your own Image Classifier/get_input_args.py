#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: Jitesh G
# DATE CREATED: 02/03/2020                            
# REVISED DATE: 
# PURPOSE: Function that retrieves the following  command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     Mandatory Arguments
#     1. data_directory as data_dir with default value './flowers'
#
#     Optional Arguments
#     2. Directory to save checkpoints with default value '/'
#     3. CNN Model Architecture as --arch with default value 'vgg16'
#     4. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#     5. Hyperparameters --learning_rate, --hidden_units, --epochs
#     6. Use GPU for training --gpu
#
##
# Imports python modules
import argparse
import sys

def get_input_args(mode='train'):
    """
    Retrieves and parses the  command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
     1. data_directory as --data-dir with default value '/flowers'
     2. Directory to save checkpoints with default value '/'
     3. CNN Model Architecture as --arch with default value 'vgg16'
     4. Hyperparameters --learning_rate, --hidden_units, --epochs
     6. Use GPU for training --gpu
    
    This function returns these arguments as an ArgumentParser object.
    
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    if(mode == 'train'):
        parser.add_argument('data_dir', action="store", type = str)
        parser.add_argument('--gpu', action="store_true", default = False,
                           help = 'use GPU for training')
        parser.add_argument('--arch', type = str, default ='vgg', choices=['vgg','densenet'],
                            help = 'classifier architecture to use "vgg"/"densenet"')
        parser.add_argument('--save_dir', type = str, default = None,
                            help = 'directory to save checkpoints')
        parser.add_argument('--learn_rate', type = float, default = 0.001,
                            help = 'learning rate to be applied to the training')
        parser.add_argument('--hidden_units', type = int, default = 512,
                            help = 'number of nodes in the hidden layer')
        parser.add_argument('--epochs', type = int, default = '5',
                            help = 'number of epochs to run the training for')    
    else:
        parser.add_argument('image_path', action="store", type = str)
        parser.add_argument('checkpoint', action="store", type = str)
        parser.add_argument('--top_k', type = int, default = 5,
                           help = 'return top K most likely classes')
        parser.add_argument('--category_names', type = str, default = None,
                            help = 'use a mapping of categories to real names')
        parser.add_argument('--gpu', action="store_true", default = False,
                           help = 'use GPU for inference')
    
    
    
    return parser.parse_args()
