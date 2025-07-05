import torch
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# my model
from model.basicNN import L_NN_ID as iden

# region model setting
model = iden(in_feature=4, h1=8, h2=9, out_feature=3)
torch.manual_seed = 81
epochs = 300
log_every_n_step = 10
batch_size = 32
# endregion

# region dataset
# import and modify data with pandas
directory = "data\iris\iris.csv"  # can use url
df = pd.read_csv(directory, delimiter=",")
print(df.head())

no_rep_variety = df["variety"].drop_duplicates()  # retrive none duplicated items
verbal_to_code = {
    verbal: i for i, verbal in enumerate(no_rep_variety)
}  # create verbal to code dictionary
df["variety"] = df["variety"].map(verbal_to_code)  # map/replace verbal with int

X = df.drop("variety", axis=1)  # axis=1 =remove column, axis=0 = remove 1st row/index
y = df["variety"]
# assigning inputs and labels and convert to tensor
input = torch.FloatTensor(X.values)
label = torch.LongTensor(y.values)  ##longTensor = int64

# train test split
train_input, test_input, train_label, test_label = train_test_split(
    input, label, test_size=0.2, random_state=81
)
train_dataset = TensorDataset(train_input, train_label)
test_dataset = TensorDataset(test_input, test_label)
# dataloader
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
# endregion


# region train model
# train_dataloader
trainer = L.Trainer(max_epochs=epochs, log_every_n_steps=log_every_n_step)
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

# or the following for notebook
# %reload_ext tensorboard
# %tensorboard --logdir=lightning_logs/

# endregion

# region testing
with torch.no_grad():
    best_ckpt = trainer.checkpoint_callback.best_model_path
    tester = L.Trainer()
    tester.test(model, dataloaders=test_loader, ckpt_path=best_ckpt)
    correct = 0
    for i, data in enumerate(test_input):
        test_res = torch.tensor(model.forward(test_input))
        test_res_arg = torch.argmax(test_res, dim=1, keepdim=False)
        print(f"{i + 1}.) {test_res[i]} \t {test_res_arg[i]} \t {test_label[i]}")
        if test_res_arg[i] == test_label[i]:
            correct += 1
        right = model.calculate_accuracy(test_res, test_label)
        right
    print(f"{correct} / {len(test_label)} are correct")
    print(f"accuracy = {correct / len(test_label)}")

# saving parameters
torch.save(model.state_dict(), "trained_parameters\l_id_iris.pt")

# endregion


# region Identify by model
# load model and parameters
loaded_model = iden(in_feature=4, h1=8, h2=9, out_feature=3)
loaded_model.load_state_dict(torch.load("trained_parameters\l_id_iris.pt"))

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
