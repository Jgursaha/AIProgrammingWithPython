# PROGRAMMER: Jitesh G
# DATE CREATED: 3rd March 2020                           
# REVISED DATE: 
# PURPOSE: Create a model from a provided checkpoint and predict the image class (also provided as argument)
from get_input_args import get_input_args
from load_model import load_model
from process_image import process_image
import torch
from torch import nn
import numpy as np
import json

def main():
    
    in_arg = get_input_args('predict')
    # Retrieve command line arguments
    image_path = in_arg.image_path
    checkpoint = in_arg.checkpoint
    top_k = in_arg.top_k
    category_names = in_arg.category_names
    gpu = in_arg.gpu
    
    # Load model from checkpoint
    model = load_model(checkpoint)
    criterion = nn.NLLLoss()
    
     # prepare image for prediction
    image = process_image(image_path)
    #print("in predict, printing processed image, information")
    #print(type(image), image.shape)
    
    # Predict Image  
    # Convert the numpy array to a tensor
    image_t = torch.from_numpy(image).type(torch.FloatTensor)
    #print("printing image shape after conversion to tensor")
    #print(image_t.shape)
    image_t = image_t.unsqueeze(0)
    #print("printing image shape after unsqueeze")
    #print(image_t.shape)
    
    model.eval()
    
    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    print("Device = "+str(device))
    
    
    with torch.no_grad():
        image_t = image_t.to(device)
        model.to(device)
        logps = model.forward(image_t)
        
        ps = torch.exp(logps)        
        top_p, top_class = ps.topk(top_k)
        
        top_p = top_p.cpu()
        top_p = top_p.detach().numpy().tolist()[0]
        top_class = top_class.cpu()
        top_class = top_class.detach().numpy().tolist()[0]
    
    class_to_idx = model.class_to_idx
    #print("CLASS TO IDX")
    #print(model.class_to_idx)
    #print(cat_to_name)

    # invert the class_to_idx dictionary
    idx_to_class = dict(map(reversed, class_to_idx.items()))
    #print("printing inverted idx to class dict")
    #print(idx_to_class)

    if category_names:
        
        # create label list, mapping to top K classes
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        label_list = []
        for item in top_class:

            # retrieve class from index
            classidx = idx_to_class.get(item)


            # retrieve class label
            label_list.append(cat_to_name.get(classidx))
            
        # display results
        print("Top probabilities with corresponding class labels...")
        for i in range(len(top_p)):
            print(str(top_p[i]) + "    " + label_list[i])
     
    else:
        
        print("Top probabilities with corresponding class index...")
        for i in range(len(top_class)):
              print( str(top_p[i]) + "    " + str(idx_to_class.get(top_class[i])) )
             
            
    
    
    
    
    
    
if __name__ == "__main__":
    main()