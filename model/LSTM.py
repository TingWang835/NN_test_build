import torch
import torch.nn as nn
import lightning as L
from torch.optim import Adam
from torchinfo import summary


# region Class: L_LSTM_MSE, Module: lightning, opt:adam, loss Function:MSEloss
class L_LSTM_MSE(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # batch_first = true bc dataloader sets batch_size as dimention 1
        self.fc = nn.Linear(hidden_size, output_size)
        self.learning_rate = 0.001

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        criterion = nn.MSELoss()
        loss = criterion(output_i, label_i)
        self.log("train_loss", loss)  # logging loss
        return loss

    def test_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        criterion = nn.MSELoss()
        loss = criterion(output_i, label_i)
        self.log("test_loss", loss)  # logging loss
        return loss


if __name__ == "__main__":
    model = L_LSTM_MSE(input_size=14, hidden_size=20, num_layers=1, output_size=2)
    summary(model, input_size=(32, 8, 14))
    # input.shape = (batch_size, time_steps, input_size)

# endregion
