from PIL import Image
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    width, height = im.size
    
    # resize but retain aspect ratio
    if width > height:
        a_ratio = width/height
        im.thumbnail((a_ratio * 256, 256))
    else:
        a_ratio = height/width
        im.thumbnail((256, a_ratio*256))
    
    d_width, d_height = im.size
    
    #print("printing updated image size before crop")
    #print(im.size[0], im.size[1])
    
    left = (d_width - 224)/2
    top = (d_height - 224)/2
    right = (d_width + 224)/2
    bottom = (d_height + 224)/2
    
    #print("printing left, top, right, bottom")
    #print(left, top, right, bottom)

    cropped_image = im.crop((left, top, right, bottom))
    #print("printing cropped image size")
    #print(cropped_image.size[0],cropped_image.size[1])
    
    # convert values to floats between 0-1
    numpy_image = np.array(cropped_image)/255
    
    # normalise image
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    numpy_image = (numpy_image - img_mean)/img_std
    
    # transpose image
    numpy_image = numpy_image.transpose(2,0,1)
    
    #print("printing final image")
    #print(numpy_image.shape)
    
    return numpy_image