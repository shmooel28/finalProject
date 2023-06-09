import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Define a dictionary to map age labels to integers
age_label2int = {
    "Top soil": 0,
    "EB": 1,
    "CH": 2,
    "Hamra": 3,
}

# Load the Excel file into a Pandas DataFrame
data = pd.read_excel("data_b.xlsx")

# Create a new DataFrame with only the "EB" and "CH" rows
filtered_data = data.loc[data["Period"].isin(["EB", "CH"])]

# Convert the age labels to integers using the dictionary
labels = filtered_data["Period"].apply(lambda x: age_label2int[x])

# Extract the rest of the columns as the features (X) and drop unnecessary columns
features = filtered_data.drop("Period", axis=1)
features=features.drop(features.columns[0], axis=1)
# Convert the data types to float32
labels = labels.astype(np.float32)
features = features.astype(np.float32)

# Split the data into training, validation, and testing sets
'''X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

# Initialize the SVM model with a linear kernel
svm_model = SVC(kernel='rbf')

# Train the SVM model on the training data
svm_model.fit(X_train, y_train)

# Evaluate the SVM model on the validation set
y_pred = svm_model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='weighted')

print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
print("Validation F1 Score: {:.2f}".format(f1))

# Evaluate the SVM model on the test set
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("Test F1 Score: {:.2f}".format(f1))'''


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Define the SVM model as a subclass of nn.Module
class SVMModel(nn.Module):
    def __init__(self):
        super(SVMModel, self).__init__()
        self.fc = nn.Linear(features.shape[1], len(age_label2int))  # Linear layer for classification

    def forward(self, x):
        return self.fc(x)

def main():
    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values)
    y_train_tensor = torch.tensor(y_train.values)
    X_test_tensor = torch.tensor(X_test.values)
    y_test_tensor = torch.tensor(y_test.values)

    # Create a DataLoader for the training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the SVM model
    svm_model = SVMModel()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(svm_model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
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

if __name__ == "__main__":
    main()
