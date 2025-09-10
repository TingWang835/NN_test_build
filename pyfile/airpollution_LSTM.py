# Packages
import torch
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchmetrics.functional import r2_score
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# my model
from model.LSTM import L_LSTM_MSE as LSTM

# region model setting
model = LSTM(input_size=14, hidden_size=28, num_layers=2, output_size=2)
window_size = 72  # determined sliding window size
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
print(df.isnull().sum())
print("there are 2067 nans in pm2.5 column")

# fill nan with median to maintain time series continuity
df["pm2.5"] = df.groupby(["month", "hour"])["pm2.5"].transform(
    lambda x: x.fillna(x.median())
)
# group pm2.5 by month and hour, fill NA with the mediam of each subgroup
# skipped 'day' column bc it left 6 NAN after filling

# combine to datetime and convert to unix timestamp
df["datetime"] = pd.to_datetime(
    df["year"].astype(str)
    + "-"
    + df["month"].astype(str)
    + "-"
    + df["day"].astype(str)
    + " "
    + df["hour"].astype(str)
    + ":00"
    + ":00"
)

df = df.set_index("datetime")  # set datetime as index
df["timestamp"] = df.index.map(pd.Timestamp.timestamp)  # convert to time stamp

# encode timestamp into cyclical features
day = 60 * 60 * 24
year = 365.2425 * day  # account for leap year
df["day_sin"] = np.sin(df["timestamp"] * (2 * np.pi / day))
df["day_cos"] = np.cos(df["timestamp"] * (2 * np.pi / day))
df["year_sin"] = np.sin(df["timestamp"] * (2 * np.pi / year))
df["year_cos"] = np.cos(df["timestamp"] * (2 * np.pi / year))
df.insert(0, "day_sin", df.pop("day_sin"))  # relocate columns
df.insert(1, "day_cos", df.pop("day_cos"))
df.insert(2, "year_sin", df.pop("year_sin"))
df.insert(3, "year_cos", df.pop("year_cos"))
df.drop(["No", "year", "month", "day", "hour"], axis=1, inplace=True)
# dropping unneeded columns

# encode category data
df = pd.get_dummies(df, columns=["cbwd"], dtype=int, drop_first=True)
# create dummy columns for 'cbwd'
df.insert(8, "cbwd_NW", df.pop("cbwd_NW"))  # relocate dummy columns
df.insert(9, "cbwd_SE", df.pop("cbwd_SE"))
df.insert(10, "cbwd_cv", df.pop("cbwd_cv"))
print(df.head())

# normalization with StandardScaler
y_scaler = StandardScaler()
y_scaler.fit(df.iloc[:, np.r_[4, 6]])
# fitting a scaler for label pm2.5 and TEMP
X_scaler = StandardScaler()
df.iloc[:, np.r_[4:8, 11:14]] = X_scaler.fit_transform(df.iloc[:, np.r_[4:8, 11:14]])
# avoid transform for cols: day/year_sin/cos + cbwd dummies + timestamp


# Sliding window
def slide_window(df, window_size):
    X, y, ts = [], [], []
    for i in range(len(df) - window_size):
        window = df.iloc[i : (i + window_size), 0:14].values  # exclude timestamp column
        X.append(window)

        label = df.iloc[i + window_size, [4, 6]].values
        # taking column 4 pm2.5 and 6 TEMP as label
        y.append(label)

        timestamp_col = df.iloc[i + window_size, [14]].values
        ts.append(timestamp_col)
        # collect timestamp for converting to datetime in validation
    return np.array(X), np.array(y), np.array(ts)


X, y, ts = slide_window(df, window_size)


# convert X, y into dataloader
input = torch.FloatTensor(X)
label = torch.FloatTensor(y)

# train test split
train_input, temp_input, train_label, temp_label = train_test_split(
    input,
    label,
    test_size=0.3,
    shuffle=False,  # no shuffle for time series
    random_state=81,
)

test_input, val_input, test_label, val_label = train_test_split(
    temp_input,
    temp_label,
    test_size=0.5,
    shuffle=False,  # no shuffle for time series
    random_state=81,
)
train_dataset = TensorDataset(train_input, train_label)
test_dataset = TensorDataset(test_input, test_label)
val_dataset = TensorDataset(val_input, val_label)
# dataloader
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
# no shuffle bc time series data
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
# endregion


# region training
trainer = L.Trainer(max_epochs=epochs, log_every_n_steps=log_every_n_step)
# learning rate finder
tuner = L.pytorch.tuner.Tuner(trainer)
lr_find_results = tuner.lr_find(
    model,
    train_dataloaders=train_loader,
    min_lr=0.00001,
    max_lr=1.0,
    early_stop_threshold=None,
)
new_lr = lr_find_results.suggestion()
model.learning_rate = new_lr
print(f"lr_find() suggest {new_lr:5f} for the learning rate.")
# train to fit
trainer.fit(model, train_dataloaders=train_loader)


# path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path
# trainer = L.Trainer(max_epochs=80, log_every_n_steps=log_every_n_step)
# trainer.fit(model, train_dataloaders=train_loader, ckpt_path=path_to_best_checkpoint)

# evaluating train/loss result
# tensorboard --logdir=lightning_logs/

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
    # edgecolors="black",  # marker edge color (not for unfilled marker)
    linewidths=1,  # axis width
    alpha=0.80,  # axis alpha
)
plt.title("Label vs Prediction PM2.5")
plt.xlabel("Label PM2.5 (Normalized)")
plt.ylabel("Predicted PM2.5 (Normalized)")
plt.annotate(
    "pm2.5-r-squared = {:.3f}".format(r2_score(test_label[:, 0], test_res[:, 0])),
    (0.05, 0.45),
)
# (0,1) argument are x and y coordinate on the graph

plt.subplot(1, 2, 2)
plt.scatter(
    test_label[:, 1],
    test_res[:, 1],
    marker="o",  # marker shape
    s=30,  # marker size
    c="#fb9a99",  # marker color hex code (grapefruit)
    edgecolors="black",  # marker edge color
    linewidths=1,  # axis width
    alpha=0.80,  # axis alpha
)
plt.title("Label vs Prediction TEMP")
plt.xlabel("Label TEMP (Normalized)")
plt.ylabel("Predicted TEMP (Normalized)")
plt.annotate(
    "TEMP-r-squared = {:.3f}".format(r2_score(test_label[:, 1], test_res[:, 1])),
    (0.15, 0.9),
)
# (0,1) argument are x and y coordinate on the graph

plt.tight_layout()
plt.show()

torch.save(model.state_dict(), "trained_parameters\l_LSTM_airpollution.pt")
# endregion

# region prediction
loaded_model = LSTM(input_size=14, hidden_size=28, num_layers=2, output_size=2)
loaded_model.load_state_dict(torch.load("trained_parameters\l_LSTM_airpollution.pt"))

best_ckpt = trainer.checkpoint_callback.best_model_path
trainer.test(loaded_model, dataloaders=test_loader, ckpt_path=best_ckpt)

# val_output, val_label, datetime convert to df
val_output = loaded_model.forward(val_input).detach()
val_output = pd.DataFrame(val_output.numpy())
val_label = pd.DataFrame(val_label.numpy())
print(val_output.shape, val_label.shape)
timestamp = pd.DataFrame(ts[-6563:])  # select the last 6563 row from ts
print(timestamp.shape)
# conver timestamp to datetime
datetime = pd.to_datetime(timestamp[0], unit="s")

res = pd.concat([datetime, val_output, val_label], axis=1)
res.columns = ["datetime", "pm2.5_pred", "TEMP_pred", "pm2.5_label", "TEMP_label"]

# inverse scaler
res.iloc[:, np.r_[1:3]] = y_scaler.inverse_transform(res.iloc[:, np.r_[1:3]])
res.iloc[:, np.r_[3:5]] = y_scaler.inverse_transform(res.iloc[:, np.r_[3:5]])
# inverse scaler


# scatter plots
# pm2.5 plot
plt.figure(figsize=(10, 6))
plt.style.use("seaborn-v0_8-colorblind")
# line 1
plt.plot(
    res["datetime"],
    res["pm2.5_pred"],
    label="Prediction",
    color="#e9cf0d",
    linestyle="-",
    marker="",
    alpha=0.8,
)

# line 2
plt.plot(
    res["datetime"],
    res["pm2.5_label"],
    label="Label",
    color="#17b2dd",
    linestyle="-",
    marker="",
    alpha=0.7,
)
# show only date on x-axis
date_form = mdates.DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.gcf().autofmt_xdate()
plt.title("Label vs Predicted PM2.5")
plt.xlabel("Year (2014-2015)")
plt.ylabel("PM2.5 value")
plt.legend()

# TEMP plot
plt.figure(figsize=(10, 6))
plt.style.use("seaborn-v0_8-colorblind")
# line 1
plt.plot(
    res["datetime"],
    res["TEMP_pred"],
    label="Prediction",
    color="#27e90d",
    linestyle="-",
    marker="",
    alpha=0.8,
)

# line 2
plt.plot(
    res["datetime"],
    res["TEMP_label"],
    label="Label",
    color="#7717dd",
    linestyle="-",
    marker="",
    alpha=0.5,
)
# show only date on x-axis
date_form = mdates.DateFormatter("%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.gcf().autofmt_xdate()
plt.title("Label vs Predicted TEMP")
plt.xlabel("Year (2014-2015)")
plt.ylabel("TEMP(°C)")
plt.legend()

print(
    f"Predicted pm2.5 value is {res[-1, 1]}, temperature {res[-1, 2]}°C ,at {res[-1, 0]} in BeiJing"
)

res.to_csv("predicted_pollution.csv", index=False)
