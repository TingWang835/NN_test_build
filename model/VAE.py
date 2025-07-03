import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.optim import Adam


# region Class: L_VAE_1, Module: lightning, opt:adam, loss Function:BCEloss
class L_VAE_1(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.fc4 = nn.Linear(latent_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

        # lr
        self.learning_rate = 0.001

    def encoder(self, x):
        hid = F.relu(self.fc1(x))
        mu, sigma = self.fc2(hid), self.fc3(hid)
        return mu, sigma

    def decoder(self, z):
        hid = F.relu(self.fc4(z))
        output = torch.sigmoid(self.fc5(hid))
        # sigmoid to create binary data
        return output

    def forward(self, x):
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)
        # create a normal distribution with same dim as sigma
        z_reparameterize = mu + sigma * epsilon
        reconstruct = self.decoder(z_reparameterize)
        return reconstruct, mu, sigma

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        input_i, _ = batch
        reconstruct, mu, sigma = self.forward(input_i)
        criterion = nn.BCELoss(reduction="sum")
        recon_loss = criterion(reconstruct, input_i)
        # BCELoss for binary data like in MNIST
        kl_divergence = torch.sum(
            -0.5 * (1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        )
        loss = recon_loss + kl_divergence
        self.log("train_loss", loss)
        return loss


if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)
    vae = L_VAE_1(input_dim=784, hidden_dim=200, latent_dim=20)
    reconstruct, mu, sigma = vae(x)
    print(reconstruct.shape)
    print(mu.shape)
    print(sigma.shape)


# endregion
