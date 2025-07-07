import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.functional import r2_score


# region Class: NN_ID, Module: lightning, opt:adam, loss Function: CrossEntropyLoss
class L_NN_ID(L.LightningModule):
    def __init__(self, in_feature, h1, h2, out_feature):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_feature)
        self.learning_rate = 0.01

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def calculate_accuracy(self, output, label):
        prediction = torch.argmax(output, dim=1, keepdim=False)
        total = label.size(0)
        check = (prediction == label).sum().item()
        return check / total

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        criterion = nn.CrossEntropyLoss()  ##important: do not combine these two lines
        loss = criterion(output_i, label_i)  ##important: do not combine these two lines
        self.log("train_loss", loss)  # logging loss
        return loss

    def test_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output_i, label_i)
        return loss


# endregion


# region Class: NN_Pred, Module: lightning, opt:adam, loss Function: MSELoss
class L_NN_Pred(L.LightningModule):
    def __init__(self, in_feature, h1, h2, h3, out_feature):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_feature)
        self.learning_rate = 0.001

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        criterion = nn.MSELoss()
        loss = criterion(output_i, label_i)
        self.log("train_loss", loss)  # logging loss
        rsquare = r2_score(label_i, output_i)
        self.log("train_r^2", rsquare)  # logging R square score
        return loss

    def test_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        criterion = nn.MSELoss()  ##separate lines to be more adaptive
        loss = criterion(output_i, label_i)
        return loss


# endregion
