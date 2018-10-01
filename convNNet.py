#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:40:12 2018

@author: hruiz

Defines the standard Neural Network class using PyTorch nn-package. It defines the model and inplements the training procedure. 
INPUT: ->data; a list of (input,output) pairs for training and validation [(x_train,y_train),(x_val,y_val)]. 
               The dimensions of x and y arrays must be (Samples,Dim) and they must be torch.FloatTensor
       ->kwarg: loss='MSE'; string name of loss function that defines the problem (only default implemented)
                activation='ReLU'; name of activation function defining the non-linearity in the network (only default implemented)
                betas=(0.9, 0.999) is a tuple with beta1 and beta2 for Adam
TO DO:
    - clean up the implementation of non linearity: it can be done by defining the non-lin at the object construction and so avoid the if-clauses
    - Apply BatchNorm
    - Implement a more flexible construction of the NN (with different widths for the hidden layers)
"""

import torch
import torch.nn as nn
import numpy as np 
from Nets.cnn import cnn
#from matplotlib import pyplot as plt

class convNNet(object):
    
    def __init__(self,data, kernel_size = 5, max_pool_size = 3, 
                 c1_out = 10, c2_out = 10, hiduni = 90, BN=False):
        
        self.kernel_size = kernel_size
        self.max_pool_size = max_pool_size
        self.c1_out = c1_out
        self.c2_out = c2_out
        self.hiduni = hiduni
        
        if type(data) is not str: 
           self.x_train, self.y_train = data[0]
           self.x_val, self.y_val = data[1]
           self.D_in = self.x_train.size()[2]
           self.D_out = self.y_train.size()[1]
           self._BN = BN
           self.loss_str = 'MSE'
           self.activ = 'ReLU'
           self._tests()
        
           ################### DEFINE MODEL ######################################
           self._contruct_model()
           if isinstance(self.x_train.data,torch.cuda.FloatTensor): 
               self.itype = torch.cuda.LongTensor
               self.model.cuda()
               self.loss_fn.cuda()
               print('Sent to GPU')
           else: 
               self.itype = torch.LongTensor
            
        elif type(data) is str:
            self._load_model(data)
        else:
            assert 1==0, 'Arguments must be either data (torch.tensor) or a string to load the model!'
            
        
    def _load_model(self,data_dir):
        print('Loading the model from '+data_dir)
        state_dic = torch.load(data_dir)
        if list(filter(lambda x: 'running_mean' in x,state_dic.keys())):
            print('BN active in loaded model')
            self._BN = True
        else:
            self._BN = False
            
        self.loss_str = state_dic['loss']
        self.activ = state_dic['activation']
        print('NN loaded with activation ',self.activ,', and loss ',self.loss_str)
        
        state_dic.popitem() #Remove the last two entries of OrderedDict
        state_dic.popitem()  
        itms = list(state_dic.items())  
        layers = list(filter(lambda x: ('weight' in x[0]) and (len(x[1].shape)==2),itms))
        self.depth = len(layers)-2
        self.width = layers[0][1].shape[0]
        self.D_in = layers[0][1].shape[1]
        self.D_out = layers[-1][1].shape[0]

        self._contruct_model()
        
        self.model.load_state_dict(state_dic)

        if isinstance(layers[-1][1],torch.cuda.FloatTensor): 
            self.itype = torch.cuda.LongTensor
            self.model.cuda()
            self.loss_fn.cuda()    
        else: 
            self.itype = torch.LongTensor
        self.model.eval()
            
    def _contruct_model(self):
        
        net = cnn(self.D_in,self.kernel_size, self.max_pool_size,self.c1_out,self.c2_out,self.hiduni)
        modules = [m for m in net.modules() if m != None]
        
        print(r'Model constructed with modules: \n', modules[1:])
        self.model = net
        self.loss_fn = nn.MSELoss()
        
    def train_nn(self,learning_rate,nr_epochs,batch_size,betas=(0.9, 0.999),opt='ADAM',seed=False):   
        """TO DO: 
            check if x_train, x_val and y_train and y_val are defined, if not, raise an error asking to define
        """
        
        if seed:
            torch.manual_seed(22)
            print('The torch RNG is seeded!')
            
        if opt == 'ADAM':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas) 
            print('Prediction using ADAM optimaizer')
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate) 
            print('WARNING! Optimizer NOT specified; Prediction using SGD optimaizer')
            
            
        self.L_val = np.zeros((nr_epochs,))
        self.L_train = np.zeros((nr_epochs,))
        for epoch in range(nr_epochs):
    
            permutation = torch.randperm(self.x_train.size()[0]).type(self.itype) # Permute indices 
            running_loss = 0.0 
            nr_minibatches = 0

            for i in range(0,len(permutation),batch_size):
                
                # Forward pass: compute predicted y by passing x to the model.
                indices = permutation[i:i+batch_size]
                y_pred = self.model(self.x_train[indices])
                
                # Compute and print loss.
                loss = self.loss_fn(y_pred, self.y_train[indices])
                running_loss += loss.item()      
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()
                
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
                
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()
                nr_minibatches += 1
        
            
            self.model.eval()  
            y = self.model(self.x_val)
            loss = self.loss_fn(y, self.y_val)
            self.model.train()  
            self.L_val[epoch] = loss.item()
            self.L_train[epoch] = running_loss/nr_minibatches
            print('Epoch:',epoch,'Val. Error:', loss.item(),'Training Error:',running_loss/nr_minibatches)
            
        print('Finished Training')
#        plt.figure()
#        plt.plot(np.arange(nr_epochs),self.L_val)
#        plt.show()
        
    def _tests(self):
        if not self.x_train.size()[0] == self.y_train.size()[0]: 
            raise NameError('Input and Output Batch Sizes do not match!')
    
    def save_model(self,path):
        state_dic = self.model.state_dict()
        state_dic['activation'] = self.activ
        state_dic['loss'] = self.loss_str
        torch.save(state_dic,path)
#class Net(nn.Module):
#    
#    def __init__(self,depth,width,D_in,D_out):
#        super(Net, self).__init__()
#        self.depth = depth
#        self.l_in = nn.Linear(D_in, width)
#        self.l_out = nn.Linear(width, D_out)
#        self.l_hid = nn.Linear(width,width)
#        
#    def forward(self,x):
