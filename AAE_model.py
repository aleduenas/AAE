import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

class Adversarial_Auto_Encoder:
    def __init__(self, input_dim , hidden_dim, encoded_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoded_dim = encoded_dim
    
    # All the modules for the algorithm. 
    # First is the Autoencoder, a separate encoder and decoder both neural net with 3 hidden layers of 100 neurons.
    # A simple logistic regressor to act as the predictor on top of the encoder
    # An adversary which acts as a predictor to determine if the protected variable was used or not
    # Adversary shares the same architecture as the encoder: a neural net with 3 hidden layers of 100 units 
    # with a sigmoid activation function at the end. 

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, encoded_dim)

            self.drop = nn.Dropout(p=0.1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.drop(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = torch.relu(x)
            R = self.fc3(x)
            return R

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(encoded_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, input_dim)

            self.drop = nn.Dropout(p=0.1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.drop(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = torch.relu(x)
            Xhat = self.fc3(x)
            return Xhat


    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(encoded_dim, 1)

        def forward(self, x):
            Y_hat = torch.sigmoid(self.fc1(x))
            return Y_hat

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(encoded_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

            self.drop = nn.Dropout(p=0.1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.drop(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = torch.relu(x)
            S_hat = torch.sigmoid(self.fc3(x))
            return S_hat

    # Initiate all the modules for the algorithm with their loss function and optimizers
    
    class Model:

        def __init__(self, alpha, beta, gamma):
            super().__init__()
            
            # Loss Hyperparameters
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            
            # Modules
            self.Enc = self.Encoder()
            self.Dec = self.Decoder()
            self.Pred = self.Predictor()
            self.Adv = self.Adversary()
            
            # Optimizers
            self.adv_optimizer = torch.optim.Adam(Adv.parameters(),
                                 lr = 0.0001,
                                 weight_decay = 1e-8)

            self.pred_optimizer = torch.optim.Adam(Pred.parameters(),
                                         lr = 0.0001,
                                         weight_decay = 1e-8)

            self.enc_optimizer = torch.optim.Adam(Enc.parameters(),
                                         lr = 0.0001,
                                         weight_decay = 1e-8)
            self.dec_optimizer = torch.optim.Adam(Dec.parameters(),
                                         lr = 0.0001,
                                         weight_decay = 1e-8)
            
            # Loss functions
            self.reconstruction_loss = nn.MSELoss()

            self.prediction_loss = nn.BCELoss()

            self.adversary_loss = nn.BCELoss()    
    # define training 
    
    def train_model(self, epoch, dataloader, Model):
        
        
        # Training time
        for epoch in range(epochs):
            # Alternating gradient
            U = 1
            
            # X is input
            # Y is label
            # S is protected variable
            for X, Y, S in dataloader:
                
                # Feeding the data
                R = Model.Enc(X)
                X_hat = Model.Dec(R)
                Y_hat = Model.Pred(R)
                S_hat = Model.Adv(R)
                
                # loss for the reconstruction
                C = Model.reconstruction_loss(X_hat, X)

                # loss for the predictor
                E = Model.prediction_loss(Y_hat, Y.unsqueeze(1))

                # loss for adversary
                D = -Model.adversary_loss(S_hat, S.unsqueeze(1))

                # Joint loss 
                L = alpha*C + beta*D + gamma*E

                # Gradient steps
                L.backward()
                Model.enc_optimizer.step()
                Model.dec_optimizer.step()

                if U:
                    # Updating Predictor
                    Model.pred_optimizer.step()
                    Model.pred_optimizer.zero_grad()

                else:
                    # Updating Adversary
                    Model.adv_optimizer.step()
                    Model.adv_optimizer.zero_grad()


                Model.enc_optimizer.zero_grad()
                Model.dec_optimizer.zero_grad()

                #alternative gradient step 
                U = not U
                
        return Model
    
    def evaluate_model(self, testloader, Model):
        # Set the model to evaluation mode
        Model.Enc.eval()
        Model.Pred.eval()

        pred_female = 0
        true_female = 0 
        correct = 0
        total = 0

        # Iterate over the test set
        with torch.no_grad():
            for inputs, labels, protected  in test_loader:
                encoding = Model.Enc(inputs)
                predictions = Model.Pred(encoding).squeeze(1) > 0.5
                labels = labels > 0.5
                total += labels.size(0)
                results = (predictions == labels)
                correct += results.sum()

                # Tracking prediction ratio for female, male is 1-female
                pred_female += predictions.sum()
                true_female += labels.sum()
        
        # Getting Y_delta
        pred_male = total - pred_female
        true_male = total - true_female      
        y_disc = abs(pred_male/true_male - pred_female/true_female)
        y_acc = correct / total
        y_t_delta = y_acc - t*y_disc
        
        results = [y_t_delta, y_disc, y_acc]
        
        return results