Purpose of the repository:
This repository is an exhibition of the machine learning models I have programed using popular datasets. Despite being a junior level self taught programmer, the models shown here did have readability, maintainability and scalability in mind when programmed, so that incase of need they can be quickly fitted on other datasets; or used by others thereof. Still I am aware that my programming skills are far from perfect, so please feel free to give me pointers :)

Structure:
data : raw data files (.csv) are stored
model: trained parameters of various models are stored
python: .py files are stored, each has matching name with their jupyter notebook version.

Current trained models includes:

    1. Iris_identification_lightning.ipynb: Pytorch lightning nn for identification (using iris dataset (1988))


Failed model
Prediction model for Diamonds dataset, failed possiblly bc there are 3 categorical columns which did not provide sufficient learning material to predict a price variation between $326~18823, despite the sheer size of the dataset.
