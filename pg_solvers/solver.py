from datetime import datetime
import os
import timeit

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optimizer=Adam, optim_args={},
                 loss_func=MSELoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optimizer = optimizer
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_loss_epoch_history = []
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
        log_dir = os.path.join('../pg_models', 'runs', datetime.now().strftime('%b%d_%H-%M-%S'))
        writer = SummaryWriter(logdir=log_dir, comment='train')

        if verbose: print('START TRAIN.')
        start_time = timeit.default_timer()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                data = data.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass to get outputs
                out = model(data)
                
                # calculate the loss between predicted and target keypoints
                out_flatten = out.flatten()
                y_flatten = data.y.flatten()
                train_loss = self.loss_func(out_flatten, y_flatten)
                writer.add_scalar('data', train_loss.item(), epoch)
                self.train_loss_history.append(train_loss)
                
                # backward pass to calculate the weight gradients
                train_loss.backward()

                # update the weights
                optimizer.step()

                # log the loss every log_nth iterations
                running_loss += train_loss.item()
                if i % log_nth == log_nth - 1:
                    if verbose:
                        print('[Iteration %d/%d] loss: %.3f' 
                              %(i + 1, len(train_loader), running_loss / log_nth))
                    running_loss = 0.0

            # store loss for each batch
            train_loss_epoch = np.mean([x.detach().cpu().numpy() 
                                        for x in self.train_loss_history[-i-1:]])
            val_loss = self.val(model, val_loader, self.loss_func)
            
            writer.add_scalars('data', {'train': train_loss_epoch, 'val': val_loss}, epoch)
            
            self.train_loss_epoch_history.append(train_loss_epoch)
            self.val_loss_history.append(val_loss)

            if verbose:
                print('[Epoch %d/%d] train_loss: %.5f - val_loss: %.5f'
                      %(epoch + 1, num_epochs, train_loss_epoch, val_loss))

        if verbose: print('FINISH.')

    def val(self, model, val_loader, loss_func, verbose=False):
        
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
