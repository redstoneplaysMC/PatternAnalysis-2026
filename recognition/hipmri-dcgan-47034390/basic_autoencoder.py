"""
Basic Autoencoder implementation for DCGAN recognition.
"""
import torch
import torch.nn as nn
class BasicAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BasicAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    def reconstruct(self, x):
        return self.forward(x)
    def latent_dim(self):
        return self.encoder[0].out_features
    def input_dim(self):
        return self.decoder[0].out_features
    def hidden_dim(self):
        return self.encoder[0].out_features
    