import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# region dataset
# import and modify data with pandas
directory = "data\diamonds\diamonds.csv"  # can use url
df = pd.read_csv(directory, delimiter=",")
print(df.head())

##building dict for verbal data
no_rep_cut = df["cut"].drop_duplicates()  # retrive none duplicated items
v2c_cut = {
    verbal: i for i, verbal in enumerate(no_rep_cut)
}  # create verbal to code dictionary for column "cut"
print(v2c_cut)
no_rep_color = df["color"].drop_duplicates()  # retrive none duplicated items
v2c_color = {
    verbal: i for i, verbal in enumerate(no_rep_color)
}  # create verbal to code dictionary for column "color"
print(v2c_color)
no_rep_clarity = df["clarity"].drop_duplicates()  # retrive none duplicated items
v2c_clarity = {
    verbal: i for i, verbal in enumerate(no_rep_clarity)
}  # create verbal to code dictionary for column "clarity"
print(v2c_clarity)

#####temp data for faster loading
directory = "data\diamonds\diamonds_100.csv"  # can use url
df = pd.read_csv(directory, delimiter=",")
print(df.head())

##substitute verbal with codes
df["cut"] = df["cut"].map(v2c_cut)  # map/replace verbal with int
df["color"] = df["color"].map(v2c_color)
df["clarity"] = df["clarity"].map(v2c_clarity)
print(df.head())
X = df.drop(["price"], axis=1)  # axis=1 =remove column, axis=0 = remove 1st row/index
y = df["price"]
# assigning inputs and labels and convert to tensor
input = torch.FloatTensor(X.values)
label = torch.FloatTensor(y.values)

# train test split
train_input, test_input, train_label, test_label = train_test_split(
    input, label, test_size=0.2, random_state=81
)
train_dataset = TensorDataset(train_input, train_label)
test_dataset = TensorDataset(test_input, test_label)
# dataloader
batch_size = 64  # train faster
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
# endregion


# region model
class L_prediction(L.LightningModule):
    def __init__(self, in_feature=9, h1=18, h2=19, out_feature=1):
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
        total = label.size(0)
        correct = (output == label).sum().item()
        return correct / total

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = F.mse_loss(output_i, label_i)  # must use loss functions
        self.log("train_loss", loss)  # logging loss
        return loss

    def test_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = F.mse_loss(output_i, label_i)
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
    min_lr=0.01,
    max_lr=1.0,
    early_stop_threshold=None,
)
new_lr = lr_find_results.suggestion()
model.learning_rate = new_lr
print(f"lr_find() suggest {new_lr:5f} for the learning rate.")
# train to fit
trainer.fit(model, train_dataloaders=train_loader)

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
        test_res_arg = torch.argmax(test_res, dim=1, keepdim=False)
        print(f"{i + 1}.) {test_res[i]} \t {test_res_arg[i]} \t {test_label[i]}")
        if test_res_arg[i] == test_label[i]:
            correct += 1
        accuracy = model.calculate_accuracy(test_res, test_label)
    print(f"{correct} / {len(test_label)} are correct")
    print(f"accuracy = {accuracy * 100}%")

# saving model and parameters
torch.save(model.state_dict(), "model\l_id_iris.pt")

# endregion


# region Identify by model
# load model and parameters
loaded_model = L_prediction()
loaded_model.load_state_dict(torch.load("model\l_id_iris.pt"))

# identify
with torch.no_grad():
    sus_input = torch.tensor(
        [[5, 3.5, 1.6, 0.6], [5.1, 2.5, 3, 1.1], [6.9, 3.2, 5.7, 2.3]]
    )
    print("the iris types are:")
    for i, data in enumerate(sus_input):
        sus_res = loaded_model.forward(sus_input).clone().detach()
        sus_res_arg = torch.argmax(sus_res, dim=1, keepdim=False)
        if sus_res_arg[i] == 0:
            sus = "Setosa"
        elif sus_res_arg[i] == 1:
            sus = "Versicolor"
        else:
            sus = "Virginica"
        print(f"{i + 1}.) {sus}")

# endregion
