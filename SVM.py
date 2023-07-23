import numpy as np
import pandas as pd
from sklearn.svm import SVC
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
features=features.drop(features.columns[0], axis=1)
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

# Initialize the SVM model with a linear kernel
svm_model = SVC(kernel='linear')

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
print("Test F1 Score: {:.2f}".format(f1))

model_filename = "svm_model.sav"
joblib.dump(svm_model, model_filename)