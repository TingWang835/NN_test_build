# Packages
import torch
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torchmetrics.functional import r2_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# my model
from model.LSTM import L_LSTM_MSE as LSTM

# region model setting
model = LSTM(input_size=14, hidden_size=20, num_layers=2, output_size=2)
torch.manual_seed = 81
epochs = 50
batch_size = 32
log_every_n_step = 10
# endregion

# region dataset
# import and modify data with pandas
directory = "data\pollution\AirPollution.csv"  # can use url
df = pd.read_csv(directory, delimiter=",")
print(df.head())

# convert to datetime then to separate columns
# df['date_column'] = pd.to_datetime(df['date_String'], format='%d-%m-%Y')
# df['year'] = df['date_column'].dt.year
# df['month'] = df['date_column'].dt.month
# df['day'] = df['date_column'].dt.day
# df['hour'] = df['date_column'].dt.hour

# remove NA data
df = df.dropna()  # remove any row that contains NAN
print(f"{df.shape[0]} rows left after removing NA")  # 41757 rows left

# drop number column, add pm2.5 and temp as label
df = df.drop(["No"], axis=1)
df[["pm2.5_label", "TEMP_label"]] = df[["pm2.5", "TEMP"]].shift(periods=-1)
# create labels by shifting 1 row up

# normalization with MinMaxScaler
scaler = MinMaxScaler()
df.iloc[:, np.r_[0:8, 9:14]] = scaler.fit_transform(df.iloc[:, np.r_[0:8, 9:14]])
print(df.head())
# normalize columns except 'cbwd'
# columns with dtype int64 gives warning when normalized

# encode category data
df = pd.get_dummies(df, columns=["cbwd"], dtype=int, drop_first=True)
# create dummy columns for 'cbwd'
df.insert(8, "cbwd_NW", df.pop("cbwd_NW"))  # relocate dummy columns
df.insert(9, "cbwd_SE", df.pop("cbwd_SE"))
df.insert(10, "cbwd_cv", df.pop("cbwd_cv"))
print(df.head())

# assign the last 57 row for prediction
prediction = df.tail(57)
df.drop(df.tail(57).index, inplace=True)  # dropping the last 57 row

# separate inputs and labels
X = df.drop(
    ["pm2.5_label", "TEMP_label"], axis=1
)  # axis=1 =remove column, axis=0 = remove 1st row/index
y = df[["pm2.5_label", "TEMP_label"]]
# assigning inputs and labels and convert to tensor
input = torch.FloatTensor(X.values)
label = torch.FloatTensor(y.values)

# train test split
train_input, test_input, train_label, test_label = train_test_split(
    input,
    label,
    test_size=0.1,
    shuffle=False,  # no shuffle for time series
    random_state=81,
)
train_dataset = TensorDataset(train_input, train_label)
test_dataset = TensorDataset(test_input, test_label)
# dataloader
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
# no shuffle bc time series data
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
# endregion


# region training
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
# evaluating train/loss result
# tensorboard --logdir=lightning_logs/

path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path
trainer = L.Trainer(max_epochs=100, log_every_n_steps=log_every_n_step)
trainer.fit(model, train_dataloaders=train_loader, ckpt_path=path_to_best_checkpoint)
# endregion


# region testing
best_ckpt = trainer.checkpoint_callback.best_model_path
trainer.test(model, dataloaders=test_loader, ckpt_path=best_ckpt)

# r2-score and plot
test_res = model.forward(test_input).detach()

plt.figure(figsize=(10, 5))
plt.style.use("seaborn-v0_8-colorblind")
plt.subplot(1, 2, 1)
plt.scatter(
    test_label[:, 0],
    test_res[:, 0],
    marker="x",  # marker shape
    s=30,  # marker size
    c="#1f78b4",  # marker color hex code (blue)
    # edgecolors="black",  # axis color (not for unfilled marker)
    linewidths=1,  # axis width
    alpha=0.80,  # axis alpha
)
plt.title("Label vs Prediction PM2.5")
plt.xlabel("Label PM2.5 (Normalized)")
plt.ylabel("Predicted PM2.5 (Normalized)")
plt.annotate(
    "pm2.5-r-squared = {:.3f}".format(r2_score(test_label[:, 0], test_res[:, 0])),
    (0.05, 0.59),
)
# (0,1) argument are x and y coordinate on the graph

plt.subplot(1, 2, 2)
plt.scatter(
    test_label[:, 1],
    test_res[:, 1],
    marker="o",  # marker shape
    s=30,  # marker size
    c="#fb9a99",  # marker color hex code (grapefruit)
    edgecolors="black",  # axis color
    linewidths=1,  # axis width
    alpha=0.80,  # axis alpha
)
plt.title("Label vs Prediction TEMP")
plt.xlabel("Label TEMP (Normalized)")
plt.ylabel("Predicted TEMP (Normalized)")
plt.annotate(
    "TEMP-r-squared = {:.3f}".format(r2_score(test_label[:, 1], test_res[:, 1])),
    (0.15, 1),
)
# (0,1) argument are x and y coordinate on the graph

plt.tight_layout()
plt.show()

torch.save(model.state_dict(), "trained_parameters\l_LSTM_airpollution.pt")
# endregion

# region prediction
loaded_model = LSTM(input_size=14, hidden_size=20, num_layers=2, output_size=2)
loaded_model.load_state_dict(torch.load("trained_parameters\l_LSTM_airpollution.pt"))

print(prediction)
X_real = prediction.drop(["pm2.5_label", "TEMP_label"], axis=1)
input_real = torch.FloatTensor(X_real.values)

output_real = loaded_model.forward(input_real).detach()
X_real["pm2.5_pred"] = output_real[:, 0]
X_real["TEMP_pred"] = output_real[:, 1]

cbwd = pd.from_dummies(
    X_real[["cbwd_NW", "cbwd_SE", "cbwd_cv"]], sep="_", default_category="NE"
)
# reverse dummy columns
df_pred = X_real.drop(["cbwd_NW", "cbwd_SE", "cbwd_cv"], axis=1)
df_pred.insert(8, "cbwd", cbwd)
df_pred.iloc[:, np.r_[0:8, 9:14]] = scaler.inverse_transform(
    df_pred.iloc[:, np.r_[0:8, 9:14]]
)
# inverse scaler
print(df_pred)

df_pred.to_csv("predicted_pollution.csv", index=False)
