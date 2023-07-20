import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


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
filtered_data = data.loc[data["Period"].isin(["EB", "CH","Top soil"])]

# Convert the age labels to integers using the dictionary
labels = filtered_data["Period"].apply(lambda x: age_label2int[x])

# Extract the rest of the columns as the features (X) and drop unnecessary columns
features = filtered_data.drop("Period", axis=1)
features=features.drop(features.columns[0], axis=1)


labels = labels.astype(np.float32)
features = features.astype(np.float32)
features = features.values.reshape(features.shape[0], -1)

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

# Train SVM model
svc = SVC(kernel='linear', C=1.0, random_state=42)

# Train Decision Tree model
dtc = DecisionTreeClassifier(max_depth=10, random_state=42)

# Train Random Forest model with SVM and Decision Tree as base estimators
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('svm', svc), ('dt', dtc), ('rf', rfc)],
    voting='hard'
)

# Fit the Voting Classifier on the training data
voting_clf.fit(X_train, y_train)

# Evaluate the Voting Classifier on the validation set
y_pred = voting_clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='weighted')

print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
print("Validation F1 Score: {:.2f}".format(f1))

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:\n", cm)