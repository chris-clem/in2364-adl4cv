"""Solver class for training a network."""

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

import src.config as cfg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optimizer=Adam, optim_args={}, L2_loss=MSELoss(), L1_loss=L1Loss()):
        """Constrcutor for solver class.

        Parameters
        ----------
        optimizer : torch.optim
            Optimizer to use (default: Adam)
        optim_args : dict
            Arguments for optimizer which are merged with default_adam_args
        L2_loss : torch.nn Loss
            L2 loss function to use (default: MSELoss)
        L1_loss : torch.nn Loss
            L1 loss function to use (default: L1Loss), only used as comparision 
        """
        
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optimizer = optimizer
        self.L1_loss = L1_loss
        self.L2_loss = L2_loss

        self._reset_histories()

    def _reset_histories(self):
        """ Resets train and val histories for the losses."""
        
        self.loss = {'translation_loss_L1': [], 'translation_loss_L2': [], 
                     'magnitude_loss_L1': [], 'magnitude_loss_L2': [],
                     'angle_loss_L1': [], 'angle_loss_L2': []}
        
        self.loss_epoch_history = {key: [] for key in list(self.loss.keys())}
        
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, 
              num_epochs=10, log_nth=0, verbose=False):
        """Train a given model with the provided data.

        Parameters
        ----------
        model : torch.nn.Module
            Model object initialized from a torch.nn.Module
        train_loader : torch.utils.data.DataLoader
            Train data in torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
            Val data in torch.utils.data.DataLoader
        num_epochs : int
            Total number of training epochs
        log_nth : int
            Log training accuracy and loss every nth iteration
        """
        
        optimizer = self.optimizer(model.parameters(), **self.optim_args)
        self._reset_histories()

        model.double()
        model.to(device)
    
        # Prepare the net for training
        model.train()
        
        # Logging into Tensorboard
        log_dir = os.path.join('pg_models', 'runs', 
                               datetime.now().strftime('%b%d_%H-%M-%S'))
        writer = SummaryWriter(logdir=log_dir, comment='train')

        if verbose: print('START TRAIN.')

        start_time = timeit.default_timer()
        datetime_now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        
        best_val_loss = 1e10
        
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
                
                # backward pass to calculate the weight gradients
                self.loss['translation_loss_L2'][-1].backward()

                # update the weights
                optimizer.step()

                # log the loss every log_nth iterations
                running_loss += self.loss['translation_loss_L2'][-1].item()
                
                #Compute magnitude angle metrics
                #magnitude_loss_L1, angle_loss_L1 = self._angles_magnitude_metric(out_flatten, y_flatten, 
                #self.L1_loss, rounded=False)
                #magnitude_loss_L2, angle_loss_L2 = self._angles_magnitude_metric(out_flatten, y_flatten, 
                #self.L2_loss, rounded=False)
                
                #self.loss['magnitude_loss_L1'].append(magnitude_loss_L1)
                #self.loss['angle_loss_L1'].append(angle_loss_L1)
                #self.loss['magnitude_loss_L2'].append(magnitude_loss_L2)
                #self.loss['angle_loss_L2'].append(angle_loss_L2)
                
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
            
            if val_loss < best_val_loss:
                model_name = '{}_{}_{}_{}_{}_{}_best_model.pth'.format(datetime_now,
                                                                       cfg.AUGMENTATION_COUNT,
                                                                       cfg.LAYER, cfg.K, cfg.NUM_TRAIN_SEQUENCES,
                                                                       cfg.LEARNING_RATE)
                model_path = os.path.join('pg_models', model_name)
                torch.save(model.state_dict(), model_path)
                best_val_loss = val_loss

            #writer.add_scalars('loss_data', {'train': train_loss_L2_epoch, 'val': val_loss}, epoch)
            #writer.add_scalars('metrics L1', {'magnitude': magnitude_loss_L1_epoch, 
            #                                  'angle': angle_loss_L1_epoch}, epoch)
            #writer.add_scalars('metrics L2', {'magnitude': magnitude_loss_L2_epoch, 
            #                                  'angle': angle_loss_L2_epoch}, epoch)

            self.val_loss_history.append(val_loss)
            
            if verbose:
                print('[Epoch %d/%d] train_loss: %.5f - val_loss: %.5f'
                      %(epoch + 1, num_epochs, self.loss_epoch_history['translation_loss_L2'][-1], val_loss))

        if verbose: print('FINISH.')
    
    
    def _angles_magnitude_metric(self, predicted_translation, gt_translation, loss, rounded=False):
        """Calculates loss for magnitudes of translations and for rotations of translations."""
        
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
    
    def _val(self, model, val_loader, loss_func, verbose=False):
        """Validate a given model with the provided data.

        Parameters
        ----------
        model : torch.nn.Module
            Model object initialized from a torch.nn.Module
        val_loader : torch.utils.data.DataLoader
            Val data in torch.utils.data.DataLoader
        loss_func : torch.nn Loss
            Loss function to use
        """
        
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