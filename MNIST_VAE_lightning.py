# Packages
import torch
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

# Mymodel
from model.VAE import L_VAE_BCE as vae

# region model setting
model = vae(input_dim=784, hidden_dim=200, latent_dim=20)
batch_size = 32
epochs = 50
log_every_n_step = 10
# end region


# region convert to binary
column_names = ["label"] + [f"pixel{i}" for i in range(784)]

grey_train = pd.read_csv(
    "data\MNIST\mnist_train.csv", delimiter=",", names=column_names
)
grey_test = pd.read_csv("data\MNIST\mnist_test.csv", delimiter=",", names=column_names)

pixel_train = grey_train.drop(["label"], axis=1)  # avoid labels to be converted
pixel_test = grey_test.drop(["label"], axis=1)

binary_train = (pixel_train > 128).astype(int)
binary_test = (pixel_test > 128).astype(int)

binary_train = pd.concat([grey_train["label"], binary_train], axis=1, ignore_index=True)
binary_test = pd.concat([grey_test["label"], binary_test], axis=1, ignore_index=True)

binary_train.to_csv(
    "data\MNIST\mnist_binary_train.csv", index=False, header=column_names
)
binary_test.to_csv("data\MNIST\mnist_binary_test.csv", index=False, header=column_names)
# endregion

# region dataset
# import binary MNIST data
train_data = pd.read_csv("data\MNIST\mnist_binary_train.csv", delimiter=",")
test_data = pd.read_csv("data\MNIST\mnist_binary_test.csv", delimiter=",")

train_X = train_data.drop(["label"], axis=1)
train_y = train_data["label"]
test_X = test_data.drop(["label"], axis=1)
test_y = test_data["label"]

train_input = torch.FloatTensor(train_X.values)
train_label = torch.FloatTensor(train_y.values)
test_input = torch.FloatTensor(test_X.values)
test_label = torch.FloatTensor(test_y.values)

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
    min_lr=0.005,
    max_lr=1.0,
    early_stop_threshold=None,
)
new_lr = lr_find_results.suggestion()
model.learning_rate = new_lr
print(f"lr_find() suggest {new_lr:5f} for the learning rate.")
# train to fit
trainer.fit(model, train_dataloaders=train_loader)

# continue training (if needed)

path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path
trainer = L.Trainer(max_epochs=100, log_every_n_steps=log_every_n_step)
trainer.fit(model, train_dataloaders=train_loader, ckpt_path=path_to_best_checkpoint)

# evaluating train/loss result
# tensorboard --logdir=lightning_logs/

# or the following for notebook
# %reload_ext tensorboard
# %tensorboard --logdir=lightning_logs/

torch.save(model.state_dict(), "trained_parameters\L_VAE_MNIST.pt")  # save parameters
# endregion

# region testing
# t-sne of latent space
loaded_model = vae(input_dim=784, hidden_dim=200, latent_dim=20)
loaded_model.load_state_dict(torch.load("trained_parameters\L_VAE_MNIST.pt"))


def get_latent_space(model, data_loader):
    latent_vectors = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            mu, sigma = model.encoder(inputs.view(-1, 28 * 28))
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon
            latent_vectors.append(z)
    return torch.cat(latent_vectors, dim=0).numpy()


# running t-SNE
latent_vectors = get_latent_space(loaded_model, test_loader)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca")
tsne_features = tsne.fit_transform(latent_vectors)
plt.figure(figsize=(10, 6))
plt.scatter(
    tsne_features[:, 0],
    tsne_features[:, 1],
    s=20,
    marker=".",
    c=test_label,
    cmap="Paired",
)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("2D t-SNE of the latent space")
plt.colorbar(label="Cluster Labels")
plt.show()

# endregion


# region sample and generate new from VAE
def generate_image(model, dataset, label):
    images = []
    idx = label
    for x, y in dataset:
        if y == idx:
            images.append(x)

    encodeings_digit = []
    idx = label
    for idx in range(10):
        with torch.no_grad():
            mu, sigma = model.encoder(images[idx].view(1, 784))
            encodeings_digit.append((mu, sigma))

    mu, sigma = encodeings_digit[label]
    epsilon = torch.randn_like(sigma)
    z = mu + sigma * epsilon
    out = model.decoder(z)
    out = torch.where(out > 0.2, 1, 0)  # convert sigmoid output into binary
    out = out.view(1, 28, 28)  # generate image with 1 channel, 28 * 28 pixels
    out_np = out.cpu().numpy()
    out_np = np.transpose(out_np, (1, 2, 0))
    plt.imshow(out_np, cmap="gray")
    plt.axis("off")  # Hide axes
    plt.show()


generate_image(loaded_model, test_dataset, 0)
