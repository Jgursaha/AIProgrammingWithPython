# PROGRAMMER: Jitesh G
# DATE CREATED: 2nd March 2020                           
# REVISED DATE: 
# PURPOSE: Train a new network on a dataset and save the model as a checkpoint.

import sys
import os
import torch
from torch import nn, optim
from get_input_args import get_input_args
from load_data import load_data
from create_model import create_model
from workspace_utils import active_session




def main():
    
    # Retrieve command line arguments
    in_arg = get_input_args('train')
    data_dir = in_arg.data_dir
    gpu = in_arg.gpu
    arch = in_arg.arch
    save_dir = in_arg.save_dir
    learn_rate = in_arg.learn_rate
    hidden_units = in_arg.hidden_units
    epochs = in_arg.epochs
    
    # Load and transform data
    trainloader, validloader, testloader, class_to_idx = load_data(data_dir)

    # Create Model
    model = create_model(arch, hidden_units)
    #model.class_to_idx = class_to_idx
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    # Execute training
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model.to(device);

    steps = 0
    running_loss = 0
    print_every = 5

    print("Training started...")
    
    with active_session():
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)

                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()

    # Do validation on the test set
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print (f"Test loss: {test_loss/len(testloader):.3f}.. "
                f"Test accuracy: {accuracy/len(testloader):.3f}")
            
    model.train()
    
    model.to("cpu")
    checkpoint = {
                'arch': arch,
                'classifier' : model.classifier,
                'learning_rate': learn_rate,
                'hidden_units': hidden_units,
                'epochs': epochs,
                'class_to_idx': class_to_idx,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
             }
    
    if save_dir:
        is_dir = os.path.isdir(save_dir)
        if not is_dir:
            os.mkdir(save_dir)
        
        path = save_dir+'/checkpoint.pth'
        torch.save(checkpoint, save_dir+'/checkpoint.pth')
    else:
        torch.save(checkpoint, 'checkpoint.pth')

    
if __name__ == "__main__":
    main()
    