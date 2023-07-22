import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt


# Define a dictionary to map age labels to integers
age_label2int = {
    "Top soil": 0,
    "EB": 1,
    "CH": 2,
    "Hamra": 3,
}
age_label2int = {
    "EB": 0,
    "CH": 1,
    "Top soil": 2,

}

# Define the SVM model as a subclass of nn.Module
class SVMModel(nn.Module):
    def __init__(self):
        super(SVMModel, self).__init__()
        #self.fc = nn.Linear(, 256)  # First hidden layer
        self.fc1 = nn.Linear(features.shape[1], 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, len(age_label2int))  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values)
    y_train_tensor = torch.tensor(y_train.values)
    X_test_tensor = torch.tensor(X_test.values)
    y_test_tensor = torch.tensor(y_test.values)

    # Create a DataLoader for the training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Initialize the SVM model
    svm_model = SVMModel()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.SGD(svm_model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-----------------------")
        for inputs, labels in train_dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = svm_model(inputs.float())
            loss = criterion(outputs, labels.long())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Test the SVM model
        with torch.no_grad():
            # Forward pass on the test data
            outputs = svm_model(X_test_tensor.float())
            _, predicted = torch.max(outputs.data, 1)

            # Calculate accuracy and F1 score
            accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
            f1 = f1_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')

            print("Test Accuracy: {:.2f}%".format(accuracy * 100))
            print("Test F1 Score: {:.2f}".format(f1))
    torch.save(svm_model.state_dict(), "svm_model.pt")

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
    for _ in range(times_run):
        sum_test_acc += train(labels)
    avg_acc = sum_test_acc / times_run
    print("The average accuracy is: {:.2f}%".format(avg_acc * 100))

