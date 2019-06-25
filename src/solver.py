from datetime import datetime
import os
import timeit

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from torch.autograd import Variable
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def iou_numpy(outputs: np.array, labels: np.array):
    
    #intersection = (outputs & labels).sum((1, 2))
    #union = (outputs | labels).sum((1, 2))
    #thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10 
    #print(outputs.shape, labels.shape)
    for i in range(16):
        #print(np.sum(outputs[i]), np.sum(labels[i]))
        intersection = np.sum(np.where(np.logical_and(outputs[i]==1, labels[i]==1), 1, 0))
        union = np.sum(np.where(np.logical_or(outputs[i]==1, labels[i]==1), 1, 0))
        iou = intersection / (union+1e-8)
        #print('\t', iou)
        
        
    intersection = np.sum(np.where(np.logical_and(outputs==1, labels==1), 1, 0))
    union = np.sum(np.where(np.logical_or(outputs==1, labels==1), 1, 0))

    iou = (intersection) / (union+1e-8)
    
    return iou


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optimizer=Adam, optim_args={},
                 L2_loss=MSELoss(), L1_loss=L1Loss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optimizer = optimizer
        self.L1_loss = L1_loss
        self.L2_loss = L2_loss

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.loss = {'translation_loss_L1': [], 'translation_loss_L2': [], 'translation_loss_L1_rounded': [],
                     'translation_loss_L2_rounded': [], 'magnitude_loss_L1': [], 'magnitude_loss_L2': [],
                     'angle_loss_L1': [], 'angle_loss_L2': [], 'magnitude_loss_L1_rounded': [],
                     'magnitude_loss_L2_rounded': [], 'angle_loss_L1_rounded': [], 'angle_loss_L2_rounded': []}
        
        self.loss_epoch_history = {key: [] for key in list(self.loss.keys())}
        
        self.val_loss_history = []

    def train(self, model, 
              train_loader, val_loader, 
              num_epochs=10, log_nth=0, verbose=False):
        """
        Train a given model with the provided data.
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optimizer = self.optimizer(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        model.double()
        model.to(device)
    
        # prepare the net for training
        model.train()
        
        # Logging into Tensorboard
        log_dir = os.path.join('pg_models', 'runs', datetime.now().strftime('%b%d_%H-%M-%S'))
        writer = SummaryWriter(logdir=log_dir, comment='train')

        if verbose: print('START TRAIN.')
        start_time = timeit.default_timer()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                
                data = data.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass to get outputs
                out = model(data)

                # calculate the loss between predicted and target keypoints
                out_flatten = out.flatten()
                y_flatten = data.y.flatten()
                
                self.loss['translation_loss_L1'].append(self.L1_loss(out_flatten, y_flatten))
                self.loss['translation_loss_L2'].append(self.L2_loss(out_flatten, y_flatten))             
                self.loss['translation_loss_L1_rounded'].append(self.L1_loss(torch.round(out_flatten), y_flatten))
                self.loss['translation_loss_L2_rounded'].append(self.L2_loss(torch.round(out_flatten), y_flatten))  
                
                # backward pass to calculate the weight gradients
                self.loss['translation_loss_L2'][-1].backward()

                # update the weights
                optimizer.step()

                # log the loss every log_nth iterations
                running_loss += self.loss['translation_loss_L2'][-1].item()
                
                #Compute magnitude angle metrics
                #magnitude_loss_L1, angle_loss_L1 = self._angles_magnitude_metric(out_flatten, y_flatten, self.L1_loss, rounded = False)
                #magnitude_loss_L2, angle_loss_L2 = self._angles_magnitude_metric(out_flatten, y_flatten, self.L2_loss, rounded = False)
                #magnitude_loss_L1_rounded, angle_loss_L1_rounded = self._angles_magnitude_metric(out_flatten, y_flatten, self.L1_loss, rounded = True)
                #magnitude_loss_L2_rounded, angle_loss_L2_rounded = self._angles_magnitude_metric(out_flatten, y_flatten, self.L2_loss, rounded = True)
                #Compute IOU metric
                #contour = data.contour.double()
                #self._IOU_of_resulting_shapes(contour=contour, translation_pred=out, translation_gt=data.y, img_shape=data.img.shape)
                
                
                #self.loss['magnitude_loss_L1'].append(magnitude_loss_L1)
                #self.loss['angle_loss_L1'].append(angle_loss_L1)
                #self.loss['magnitude_loss_L2'].append(magnitude_loss_L2)
                #self.loss['angle_loss_L2'].append(angle_loss_L2)
                #self.loss['magnitude_loss_L1_rounded'].append(magnitude_loss_L1_rounded)
                #self.loss['angle_loss_L1_rounded'].append(angle_loss_L1_rounded)
                #self.loss['magnitude_loss_L2_rounded'].append(magnitude_loss_L2_rounded)
                #self.loss['angle_loss_L2_rounded'].append(angle_loss_L2_rounded)
                
                if i % log_nth == log_nth - 1:
                    if verbose:
                        print('[Iteration %d/%d] loss: %.3f' 
                              %(i + 1, len(train_loader), running_loss / log_nth))
                    running_loss = 0.0
                
            # store loss for each batch
            for key in list(self.loss.keys()):
                epoch_loss = np.mean([x.detach().cpu().numpy() for x in self.loss[key][-i-1:]])
                self.loss_epoch_history[key].append(epoch_loss)

            val_loss = self._val(model, val_loader, self.L2_loss)

#             writer.add_scalars('loss_data', {'train': train_loss_L2_epoch, 'val': val_loss}, epoch)
#             writer.add_scalars('metrics L1', {'magnitude': magnitude_loss_L1_epoch, 'angle': angle_loss_L1_epoch}, epoch)
#             writer.add_scalars('metrics L2', {'magnitude': magnitude_loss_L2_epoch, 'angle': angle_loss_L2_epoch}, epoch)

            self.val_loss_history.append(val_loss)
            
            if verbose:
                print('[Epoch %d/%d] train_loss: %.5f - val_loss: %.5f'
                      %(epoch + 1, num_epochs, self.loss_epoch_history['translation_loss_L2'][-1], val_loss))
                #print('\tL1 Loss: translation:', self.loss_epoch_history['translation_loss_L1'][-1],
                #              'rounded:', self.loss_epoch_history['translation_loss_L1_rounded'][-1])
                #print('\tL2 Loss: translation:', self.loss_epoch_history['translation_loss_L2'][-1],
                #              'rounded:', self.loss_epoch_history['translation_loss_L2_rounded'][-1])
                #print('\tL1 Loss:   Magnitude:', self.loss_epoch_history['magnitude_loss_L1'][-1],
                #                  'Angle:', self.loss_epoch_history['angle_loss_L1'][-1])
                #print('\t  -> rounded: Magnitude:', self.loss_epoch_history['magnitude_loss_L1_rounded'][-1],
                #                  'Angle:', self.loss_epoch_history['angle_loss_L1_rounded'][-1])
                #print('\tL2 Loss:   Magnitude:', self.loss_epoch_history['magnitude_loss_L2'][-1],
                #                  'Angle:', self.loss_epoch_history['angle_loss_L2'][-1])
                #print('\t  -> rounded: Magnitude:', self.loss_epoch_history['magnitude_loss_L2_rounded'][-1],
                #                  'Angle:', self.loss_epoch_history['angle_loss_L2_rounded'][-1])

        if verbose: print('FINISH.')
    
    def _angles_magnitude_metric(self, predicted_translation, gt_translation, loss, rounded=False):
        '''Calculates MSE Loss for magnitudes of translations and
        calculates MSE Loss for rotations of translations'''
        predicted_translation = predicted_translation.view(-1, 2)
        gt_translation = gt_translation.view(-1, 2)
        
        if rounded == True:
            predicted_translation = torch.round(predicted_translation)
            
        #Calculate magnitude of translation
        magnitude_gt = torch.norm(gt_translation, dim=1)
        magnitude_predicted = torch.norm(predicted_translation, dim=1)
        magnitude_loss = loss(magnitude_predicted, magnitude_gt)

        #Calculate rotation of translation
        alpha_gt = torch.asin(gt_translation[:,1]/(magnitude_gt+1e-8))
        alpha_predicted = torch.asin(predicted_translation[:,1]/(magnitude_predicted+1e-8))     
        alpha_loss = loss(alpha_predicted, alpha_gt) * 180 / np.pi

        return magnitude_loss, alpha_loss

    def _IOU_of_resulting_shapes(self, contour, translation_pred, translation_gt, img_shape):
        #print(translation_pred.shape, translation_gt.shape)
        contour_pred = contour + translation_pred
        contour_gt = contour + translation_gt
        
        contour_img = np.zeros((img_shape[0], img_shape[2], img_shape[3])).astype(np.uint8)
        contour_pred = np.expand_dims(contour_pred.cpu().detach().numpy().astype(np.int32), axis=0)
        contour_img_pred = cv2.fillPoly(contour_img, contour_pred, color=1)
        
        
        contour_gt = np.expand_dims(contour_gt.cpu().detach().numpy().astype(np.int32), axis=0)
        contour_img_gt = cv2.fillPoly(contour_img, contour_gt, color=1)

        iou = iou_numpy(contour_img_pred, contour_img_gt)
        #print(iou)
        
        return iou
        
        
    
    def _val(self, model, val_loader, loss_func, verbose=False):
        
        model.eval()
        running_loss = 0.0
        for i, data in enumerate(val_loader):
            
            data = data.to(device)

            # forward pass to get outputs
            with torch.no_grad():
                out = model(data)

            # calculate the loss between predicted and target keypoints
            out_flatten = out.flatten()
            y_flatten = data.y.flatten()
            val_loss = loss_func(out_flatten, y_flatten)

            # log the loss every log_nth iterations
            running_loss += val_loss.item()

        val_loss = running_loss/len(val_loader)
        
        return val_loss