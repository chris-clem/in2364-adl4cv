"""
TODOS
--> Model has to be loaded for each image --> Load model once per sequence
--> Is there preprocessing steps?

"""

import torch
import torch.nn as nn
from torchvision import transforms

import OSVOS_PyTorch.networks.vgg_osvos as vo

#from torchsummary import summary
import numpy as np
from scipy.misc import imread, imsave, imresize


def get_OSVOS_feature(key_point_positions, layer, img_path, model_path):
    """
    Function takes list of keypoints as input and outputs OSVOS feature vector for each point
    key point positions list of tuples [(x, y), ...]

    """

    #1st CONV block 1-4 --> Shape: [1, 64, 480, 854] --> Receptive field: 5
    #2nd CONV block 5-9 --> Shape: [1, 128, 240, 427] --> Receptive field: 5 * 2(pooling) + 4 = 14
    #3rd CONV block 10-16 --> Shape: [1, 256, 120, 214] --> Receptive field: 14 * 2 + 6 = 34
    #4th CONV block 17-23 --> Shape: [1, 512, 60, 107] --> Receptive field: 34 * 2 + 6 = 74
    #5th CONV block 24-30 --> Shape: [1, 512, 30, 54] --> Receptive field: 74 * 2 + 6 = 154

    #Load test image and transform it in (C, H, W)
    img = np.moveaxis(imread(img_path), 2, 0).astype(float)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    print('Image shape:', img.numpy().shape)

    #Load model
    model = vo.OSVOS(pretrained=0)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()

    #Define layer as output
    #print('\nSTAGES\n\n', len(model.stages), model.stages)
    #print('\nUPSCALE\n\n', len(model.upscale), model.upscale)
    #print('\nside_prep\n\n', len(model.side_prep), model.side_prep)
    #print('\nscore_dsn\n\n', len(model.score_dsn), model.score_dsn)
    #print('\nupscale_\n\n', len(model.upscale_), model.upscale_)
    #print('\nfuse\n\n', model.fuse)

    children = []
    for num, stage in enumerate(model.stages):
        #print(type(stage), '\n', stage, '\n\n')
        if type(stage) == torch.nn.modules.container.Sequential:
        	for child in stage.children():
        		children.append(child)

    print('Create model for feature_vector after', len(children[:layer]), 'layers...')
    new_model = nn.Sequential(*children[:layer])
    new_model = new_model.double()
    new_model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = new_model.to(device)
    img = img.to(device)
    feature_vector = new_model(img)
    print('Feature vector shape:', feature_vector.shape)

    img, feature_vector = img.cpu(), feature_vector.cpu()
    #Extract vector out of output tensor depending on keypoint location
    #Compute receptive fields for every feature vector. --> Select feature vector which receptive field center is closest to key point
    _, _, height_img, width_img = img.detach().numpy().shape
    _, _, height_fv, width_fv = feature_vector.detach().numpy().shape

    feature_vectors = []
    for key_point_position in key_point_positions:
	    x_kp, y_kp = key_point_position
	    x_fv, y_fv = round(float(x_kp) * width_fv / width_img), round(float(y_kp) * height_fv / height_img)
	    #print('X', x_kp, '-->', x_fv, '\nY',  y_kp, '-->', y_fv)
	    feature_vectors.append(feature_vector[: ,: , y_fv, x_fv])

    return feature_vectors



# key_point_positions = [(7, 167), (15, 33)]
# layer = 9
# img_path = '../DAVIS_2016/DAVIS/JPEGImages/480p/blackswan/00000.jpg'
# model_path = 'models/parent_epoch-239.pth'

# get_OSVOS_feature(key_point_positions, layer, img_path, model_path)