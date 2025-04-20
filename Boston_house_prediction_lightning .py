import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# region dataset
# import and modify data with pandas
directory = "data\Boston\BostonHousing.csv"  # can use url
df = pd.read_csv(directory, delimiter=",")
print(df.head())

# normalization with MinMaxscaler
scaler = MinMaxScaler()
df.iloc[:, np.r_[1:3, 5:14]] = scaler.fit_transform(df.iloc[:, np.r_[1:3, 5:14]])
##normalize all column except number 4 "CHAS"
print(df.head())

X = df.drop(["medv"], axis=1)  # axis=1 =remove column, axis=0 = remove 1st row/index
y = df["medv"]
# assigning inputs and labels and convert to tensor
input = torch.FloatTensor(X.values)
label = torch.FloatTensor(y.values)

# train test split
train_input, test_input, train_label, test_label = train_test_split(
    input,
    label,
    test_size=0.2,
    random_state=81,
)
train_dataset = TensorDataset(train_input, train_label)
test_dataset = TensorDataset(test_input, test_label)
# dataloader
batch_size = 32
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
# endregion


# region model
class L_prediction(L.LightningModule):
    def __init__(self, in_feature=13, h1=20, h2=13, h3=13, out_feature=1):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_feature)
        self.learning_rate = 0.0001

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)

    def calculate_accuracy(self, output, label):
        total = label.size(0)
        correct = (output == label).sum().item()
        return correct / total

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        criterion = nn.MSELoss()  ##separate lines to be more adaptive
        loss = criterion(output_i, label_i)
        self.log("train_loss", loss)  # logging loss
        accuracy = self.calculate_accuracy(output_i, label_i)
        self.log("train_accuracy", accuracy)  # logging accuracy
        return loss

    def test_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        criterion = nn.MSELoss()  ##separate lines to be more adaptive
        loss = criterion(output_i, label_i)
        return loss


# endregion

# region train model
# train_dataloader
torch.manual_seed = 81
model = L_prediction()
epochs = 300
trainer = L.Trainer(max_epochs=epochs, log_every_n_steps=10)

# learning rate finder
tuner = L.pytorch.tuner.Tuner(trainer)
lr_find_results = tuner.lr_find(
    model,
    train_dataloaders=train_loader,
    min_lr=0.0001,
    max_lr=1.0,
    early_stop_threshold=None,
)
new_lr = lr_find_results.suggestion()
model.learning_rate = new_lr
print(f"lr_find() suggest {new_lr:5f} for the learning rate.")
# train to fit
trainer.fit(model, train_dataloaders=train_loader)

# continue training
path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path
trainer = L.Trainer(max_epochs=500, log_every_n_steps=2)
trainer.fit(model, train_dataloaders=train_loader, ckpt_path=path_to_best_checkpoint)

# evaluating train/loss result
# use tensorboard in terminal
# tensorboard --logdir=lightning_logs/

# endregion

# region testing
with torch.no_grad():
    best_ckpt = trainer.checkpoint_callback.best_model_path
    tester = L.Trainer()
    tester.test(model, dataloaders=test_loader, ckpt_path=best_ckpt)
    correct = 0
    for i, data in enumerate(test_input):
        test_res = model.forward(test_input).detach()
        print(f"{i + 1}.) {test_res[i]} \t {test_label[i]}")
        if test_res[i] == test_label[i]:
            correct += 1
        accuracy = model.calculate_accuracy(test_res, test_label)
    print(f"{correct} / {len(test_label)} are correct")
    print(f"accuracy = {accuracy * 100}%")

# saving model and parameters
torch.save(model.state_dict(), "model\l_pd_BostonHousing.pt")

# endregion


# Identify by model
# load model and parameters
loaded_model = L_prediction()
loaded_model.load_state_dict(torch.load("model\l_pd_diamonds.pt"))
