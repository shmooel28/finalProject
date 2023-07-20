import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Define a dictionary to map age labels to integers
age_label2int = {
    "EB": 0,
    "CH": 1,
    "Top soil": 2,
}

# Define the KNN model as a subclass of nn.Module
class KNNModel(nn.Module):
    def __init__(self, n_neighbors):
        super(KNNModel, self).__init__()
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def forward(self, x):
        return self.knn.predict(x)

def train(labels, k_neighbors):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Initialize the KNN model
    knn_model = KNNModel(n_neighbors=k_neighbors)

    # Fit the KNN model on the training data
    knn_model.knn.fit(X_train, y_train)

    # Test the KNN model
    y_pred = knn_model(torch.tensor(X_test.values).float())

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Test Accuracy: {:.2f}%".format(accuracy * 100))
    print("Test F1 Score: {:.2f}".format(f1))

    return accuracy

if __name__ == "__main__":
    # Load the Excel file into a Pandas DataFrame
    data = pd.read_excel("data_b.xlsx")

    # Create a new DataFrame with only the "EB" and "CH" rows
    filtered_data = data.loc[data["Period"].isin(["EB", "CH", "Top soil"])]

    # Convert the age labels to integers using the dictionary
    labels = filtered_data["Period"].apply(lambda x: age_label2int[x])

    features = filtered_data.drop("Period", axis=1)
    # Extract the rest of the columns as the features (X) and drop unnecessary columns
    features = features.drop(features.columns[0], axis=1)

    # Convert the data types to float32
    labels = labels.astype(np.float32)
    features = features.astype(np.float32)
    sum_test_acc = 0
    times_run = 20
    k_neighbors = 5  # You can change the number of neighbors here
    for _ in range(times_run):
        sum_test_acc += train(labels, k_neighbors)
    avg_acc = sum_test_acc / times_run
    print("The average accuracy is: {:.2f}%".format(avg_acc * 100))
    knn_model_file = "knn_model.pkl"
    joblib.dump(KNNModel, knn_model_file)
