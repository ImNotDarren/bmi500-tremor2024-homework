import pandas as pd
import numpy as np

from process import (create_labeled_dataset, process_all_files)
from model import train_evaluate_svm

# Load the data
lbl = pd.read_csv("./labels.csv")
features = process_all_files(lbl)

# Create the labeled dataset
ds = create_labeled_dataset(lbl, features)

# Create a new column that designates "easy" cases
ds['easy'] = ds['label_raw'].isin(['absent', 'severe', 'significant'])

# Preprocess the label to 0/1
ds['label_yn'] = ds['label_yn'].apply(
    lambda label: 1 if label == 'present' else 0)

# Pull all variables starting with "Feature_" into an array
feature_columns = [col for col in ds.columns if col.startswith("Feature_")]
X1 = ds[feature_columns].values
y1 = ds["label_yn"].values

X2 = ds[ds["easy"]][feature_columns].values
y2 = ds[ds["easy"]]["label_yn"].values

# Calculate the performance of each classifier
f1score = train_evaluate_svm(X1, y1)

f1score = train_evaluate_svm(X1, y1, kernel='rbf')

f1score = train_evaluate_svm(X2, y2)

f1score = train_evaluate_svm(X2, y2, kernel='rbf')
