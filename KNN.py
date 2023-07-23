import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Define a dictionary to map age labels to integers, if you want to add more ages, you need to add them here
age_label2int = {
    "Top soil": 0,
    "EB": 1,
    "CH": 2,
    "Hamra": 3,
}

# Load the Excel file into a Pandas DataFrame - the name of the xlsx file
data = pd.read_excel("data_b.xlsx")

# Create a new DataFrame with only the ages that you want to learn, in this case we not load Hamra because we don't
# have enough data
filtered_data = data.loc[data["Period"].isin(["EB", "CH","Top soil"])]

# Convert the age labels to integers using the dictionary
labels = filtered_data["Period"].apply(lambda x: age_label2int[x])

# Extract the rest of the columns as the features (X) and drop unnecessary columns, this is with the assumption you
# don't have any not necessary columns
features = filtered_data.drop("Period", axis=1)
features = features.drop(features.columns[0], axis=1)

# Convert the data types to float32
labels = labels.astype(np.float32)
features = features.astype(np.float32)

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

# Initialize the KNN model with the desired number of neighbors (e.g., 5)
k_neighbors = 5
knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)

# Train the KNN model on the training data
knn_model.fit(X_train, y_train)

# Evaluate the KNN model on the validation set
y_pred_knn = knn_model.predict(X_val)

knn_accuracy = accuracy_score(y_val, y_pred_knn)
knn_f1 = f1_score(y_val, y_pred_knn, average='weighted')

print("Validation Accuracy (KNN): {:.2f}%".format(knn_accuracy * 100))
print("Validation F1 Score (KNN): {:.2f}".format(knn_f1))

# Evaluate the KNN model on the test set
y_pred_knn_test = knn_model.predict(X_test)

knn_accuracy_test = accuracy_score(y_test, y_pred_knn_test)
knn_f1_test = f1_score(y_test, y_pred_knn_test, average='weighted')

print("Test Accuracy (KNN): {:.2f}%".format(knn_accuracy_test * 100))
print("Test F1 Score (KNN): {:.2f}".format(knn_f1_test))

# Save the trained KNN model to a file
knn_model_filename = "knn_model.sav"
joblib.dump(knn_model, knn_model_filename)

