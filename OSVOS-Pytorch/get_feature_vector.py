
"""
TODOS
--> Model has to be loaded for each image --> Load model once per sequence
--> Is there preprocessing steps?

"""

import torch
import torch.nn as nn
from torchvision import transforms

import networks.vgg_osvos as vo

from torchsummary import summary
import numpy as np
from scipy.misc import imread, imsave, imresize

"""
class VectorGetter(object):
    def __init__(self, layer_name, MODEL):
        self.model = MODEL
        self.model.eval()
        self.to_tensor = transforms.ToTensor()
        self.layer = self.model._modules.get(layer_name)

    def get_vector(self, path):

        im_object =Image.open(path)

        t_img = Variable(
            self.normalize(self.to_tensor(self.scaler(im_object))).unsqueeze(0)
        )
        my_embedding = torch.zeros(512)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data.squeeze())

        h = self.layer.register_forward_hook(copy_data)
        self.model(t_img)
        h.remove()

        return path, my_embedding.data.numpy().astype(DTYPE)
"""

def get_OSVOS_feature(key_point_positions, layer, img_path, model_path):
    """
    Function takes list of keypoints as input and outputs OSVOS feature vector for each point
    key point positions list of tuples [(x, y), ...]

    """

    #Load test image and transform it in (C, H, W)
    test_img = np.moveaxis(imread(img_path), 2, 0).astype(float)
    test_img = np.expand_dims(test_img, axis=0)
    test_img = torch.from_numpy(test_img)
    #print(test_img)
    print('Test image shape:', test_img.shape)

    #Load model
    model = vo.OSVOS(pretrained=0)
    model = model.float()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()
    print('\nOLD MODEL')
    print(type(model))
    print(model.upscale)

    summary(model, (3, 480, 854))

    test_output = model.forward(test_img.float())
    test_output = np.squeeze(test_output[0].detach().numpy(), axis=(0,1))
    print(test_output.shape)
    imsave('/home/max/in2364-adl4cv/test.png', test_output*255)

    #Define layer as output
    children = []
    for num, child in enumerate(model.children()):
        if type(child) == torch.nn.modules.conv.Conv2d:
            #print(child)
            children.append(child)
        else:
            for idx in range(child.__len__()):
                child_child = child.__getitem__(idx)
                if type(child_child) == torch.nn.modules.container.Sequential:
                    for idx2 in range(child_child.__len__()):
                        child_child_child = child_child.__getitem__(idx2)
                        children.append(child_child_child)
                        #print(child_child_child)                       
                else:
                    children.append(child_child)
                    #print(child_child)

    print(len(children))
    #Adapted manually
    #children[0] = torch.nn.ConvTranspose2d(3, 16, (4, 4), stride=(2,2), bias=False)



    print('\nNEW MODEL')
    new_model = nn.Sequential(*children)
    print(type(new_model))
    new_model.eval()
    summary(new_model, (3, 480, 854))    
    #summary(new_model, test_img.shape)
    #test_img = np.expand_dims(test_img, axis=0)
    #print(test_img.shape)

    print(test_img.shape)
    feature_vector = new_model(test_img)
    print(feature_vector)

    #Extract vector out of output tensor depending on keypoint location

    return



key_point_positions = [(5, 10), (15, 25)]
layer = 'ReLu-20'
img_path = '/home/max/in2364-adl4cv/DAVIS_2016/DAVIS/JPEGImages/480p/blackswan/00000.jpg'
model_path = '/home/max/in2364-adl4cv/OSVOS-Pytorch/models/blackswan_epoch-249.pth'

get_OSVOS_feature(key_point_positions, layer, img_path, model_path)