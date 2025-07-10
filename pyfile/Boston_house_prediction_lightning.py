# Packages
import torch
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# my model
from model.basicNN import L_NN_Pred as pred

# region model setting
model = pred(in_feature=13, h1=26, h2=27, h3=28, out_feature=1)
torch.manual_seed = 81
epochs = 100
batch_size = 32
log_every_n_step = 10
# endregion

# region dataset
# import and modify data with pandas
directory = "data\Boston\BostonHousing.csv"  # can use url
df = pd.read_csv(directory, delimiter=",")
df = df.astype("float")
print(df.head())

# normalization with MinMaxscaler
scaler = MinMaxScaler()
df.iloc[:, np.r_[1:3, 5:14]] = scaler.fit_transform(df.iloc[:, np.r_[1:3, 5:14]])
# normalize all column except number 4 "CHAS"

X = df.drop(["medv"], axis=1)  # axis=1 =remove column, axis=0 = remove 1st row/index
y = df["medv"]
# assigning inputs and labels and convert to tensor
input = torch.FloatTensor(X.values)
label = torch.FloatTensor(y.values)
label = label.reshape(-1, 1)

print(y.head())
# transform label to dim (n, 1) for loss calculation
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
    min_lr=0.001,
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
trainer = L.Trainer(max_epochs=800, log_every_n_steps=10)
trainer.fit(model, train_dataloaders=train_loader, ckpt_path=path_to_best_checkpoint)


# evaluating train/loss result
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


# Visalizing result
train_res = model.forward(train_input).detach()
test_res = model.forward(test_input).detach()

plt.figure(figsize=(10, 5))
plt.style.use("seaborn-v0_8-colorblind")
plt.subplot(1, 2, 1)
plt.scatter(
    train_label,
    train_res,
    marker="x",  # marker shape
    s=100,  # marker size
    c="#1f78b4",  # marker color hex code (blue)
    edgecolors="black",  # axis color
    linewidths=1,  # axis width
    alpha=0.80,  # axis alpha
)
plt.title("Label vs Prediction-train")
plt.xlabel("Label Value (Normalized)")
plt.ylabel("Prediction Value (Normalized)")
plt.annotate("r-squared = {:.3f}".format(r2_score(train_label, train_res)), (0, 1))
# (0,1) argument sets location of the annotation

plt.subplot(1, 2, 2)
plt.scatter(
    test_label,
    test_res,
    marker="o",  # marker shape
    s=100,  # marker size
    c="#fb9a99",  # marker color hex code (grapefruit)
    edgecolors="black",  # axis color
    linewidths=1,  # axis width
    alpha=0.80,  # axis alpha
)
plt.title("Label vs Prediction-test")
plt.xlabel("Label Value (Normalized)")
plt.ylabel("Prediction Value (Normalized)")
plt.annotate("r-squared = {:.3f}".format(r2_score(test_label, test_res)), (0.01, 1))
# (0,1) argument sets location of the annotation

plt.tight_layout()
plt.show()

# saving model and parameters
torch.save(model.state_dict(), "trained_parameters\l_pd_BostonHousing.pt")
# endregion


# region Actual Prediction
# load model and parameters
loaded_model = pred(in_feature=13, h1=26, h2=27, h3=28, out_feature=1)
loaded_model.load_state_dict(torch.load("trained_parameters\l_pd_BostonHousing.pt"))

# load CSV dataset
directory = "data\Boston\BostonHousing_100.csv"  # randomly selected 100 data to resemble real data
df_real = pd.read_csv(directory, delimiter=",")
df_real = df_real.astype("float")
# Normalize with training scaler
df_real.iloc[:, np.r_[1:3, 5:14]] = scaler.transform(df_real.iloc[:, np.r_[1:3, 5:14]])
# convert to tensor
real_X = df_real.drop(["medv"], axis=1)
real_input = torch.FloatTensor(real_X.values)  # convert to tensor
# prediction
pred_res = loaded_model.forward(real_input).clone().detach()

# reverse scaler
pred_medv = pd.DataFrame(pred_res.numpy())  # convert tensor to df
real_X.insert(loc=13, column="medv", value=pred_medv, allow_duplicates=False)
price_pred = real_X.astype("float")
price_pred.iloc[:, np.r_[1:3, 5:14]] = scaler.inverse_transform(
    price_pred.iloc[:, np.r_[1:3, 5:14]]
)
print(price_pred.head())

# save prediction as new csv if necessary
price_pred.to_csv("Price_prediction.csv", index=False)
# endregion
