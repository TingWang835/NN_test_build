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
        self.in_2_hid = nn.Linear(input_dim, hidden_dim)
        self.hid_2_mu = nn.Linear(hidden_dim, latent_dim)
        self.hid_2_sigma = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.z_2_hid = nn.Linear(latent_dim, hidden_dim)
        self.hid_2_out = nn.Linear(hidden_dim, input_dim)

        # lr
        self.learning_rate = 0.001

    def encoder(self, x):
        hid = F.relu(self.in_2_hid(x))
        mu, sigma = self.hid_2_mu(hid), self.hid_2_sigma(hid)
        return mu, sigma

    def decoder(self, z):
        hid = F.relu(self.z_2_hid(z))
        output = torch.sigmoid(self.hid_2_out(hid))
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
        recon_loss = nn.BCELoss(reconstruct, input_i)
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
