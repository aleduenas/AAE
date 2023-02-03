import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset


class Encoder(nn.Module):
    def __init__(self, input_dim , hidden_dim, encoded_dim):
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
    def __init__(self, input_dim , hidden_dim, encoded_dim):
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
    def __init__(self, encoded_dim):
        super().__init__()
        self.fc1 = nn.Linear(encoded_dim, 1)

    def forward(self, x):
        Y_hat = torch.sigmoid(self.fc1(x))
        return Y_hat

class Adversary(nn.Module):
    def __init__(self, hidden_dim, encoded_dim):
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

class AdversarialAutoEncoder:
    def __init__(self, input_dim , hidden_dim, encoded_dim, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    # All the modules for the algorithm. 
    # First is the Autoencoder, a separate encoder and decoder both neural net with 3 hidden layers of 100 neurons.
    # A simple logistic regressor to act as the predictor on top of the encoder
    # An adversary which acts as a predictor to determine if the protected variable was used or not
    # Adversary shares the same architecture as the encoder: a neural net with 3 hidden layers of 100 units 
    # with a sigmoid activation function at the end. 



    # Initiate all the modules for the algorithm with their loss function and optimizers
        # Modules
        self.encoder = Encoder(input_dim , hidden_dim, encoded_dim)
        self.decoder = Decoder(input_dim , hidden_dim, encoded_dim)
        self.predictor = Predictor(encoded_dim)
        self.adversary = Adversary(hidden_dim, encoded_dim)

        # Optimizers
        self.adv_optimizer = torch.optim.Adam(self.adversary.parameters(),
                             lr = 0.0001,
                             weight_decay = 1e-8)

        self.pred_optimizer = torch.optim.Adam(self.predictor.parameters(),
                                     lr = 0.0001,
                                     weight_decay = 1e-8)

        self.enc_optimizer = torch.optim.Adam(self.encoder.parameters(),
                                     lr = 0.0001,
                                     weight_decay = 1e-8)
        self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(),
                                     lr = 0.0001,
                                     weight_decay = 1e-8)

        # Loss functions
        self.reconstruction_loss = nn.MSELoss()

        self.prediction_loss = nn.BCELoss()

        self.adversary_loss = nn.BCELoss()    
    # define training 
    
    def train_model(self, epochs, dataloader):
        
        
        # Training time
        for epoch in range(epochs):
            # Alternating gradient
            U = 1
            
            # X is input
            # Y is label
            # S is protected variable
            for X, Y, S in dataloader:
                
                # Feeding the data
                R = self.encoder(X)
                X_hat = self.decoder(R)
                Y_hat = self.predictor(R)
                S_hat = self.adversary(R)
                
                # loss for the reconstruction
                C = self.reconstruction_loss(X_hat, X)

                # loss for the predictor
                E = self.prediction_loss(Y_hat, Y.unsqueeze(1))

                # loss for adversary
                D = -self.adversary_loss(S_hat, S.unsqueeze(1))

                # Joint loss 
                L = self.alpha*C + self.beta*D + self.gamma*E

                # Gradient steps
                L.backward()
                self.enc_optimizer.step()
                self.dec_optimizer.step()

                if U:
                    # Updating Predictor
                    self.pred_optimizer.step()
                    self.pred_optimizer.zero_grad()

                else:
                    # Updating Adversary
                    self.adv_optimizer.step()
                    self.adv_optimizer.zero_grad()


                self.enc_optimizer.zero_grad()
                self.dec_optimizer.zero_grad()

                #alternative gradient step 
                U = not U
    
    def evaluate_model(self, testloader):
        # Set the model to evaluation mode
        self.encoder.eval()
        self.predictor.eval()

        pred_female = 0
        true_female = 0 
        correct = 0
        total = 0

        # Iterate over the test set
        with torch.no_grad():
            for inputs, labels, protected  in testloader:
                encoding = self.encoder(inputs)
                predictions = self.predictor(encoding).squeeze(1) > 0.5
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
        # TODO: Add t to formula to check different t-discriminations
        y_t_delta = y_acc - y_disc
        
        results = [y_t_delta, y_disc, y_acc]
        
        return results