import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


age_label2int = {
    "Top soil": 0,
    "EB": 1,
    "CH": 2,
    "Hamra": 3,
}
char_map = {'a': '0', 'b': '1', 'c': '2'}

# Load the CSV file into a Pandas DataFrame
data = pd.read_excel("data_b.xlsx")

# Extract the Period column as the labels (Y)
labels = data["Period"]


data["Depth2"] = data["Depth2"].str.replace('|'.join(char_map.keys()), lambda x: char_map[x.group()])
for i in labels:
    if i in age_label2int:
        i = age_label2int[i]

# Extract the rest of the columns as the features (X)
data = data.drop("Period", axis=1)
features = data.drop("sample", axis=1)

# Process the Depth column to replace "a", "b", "c" with "0", "1", "2"
features["Depth"] = features["Depth"].astype(str).str.replace('|'.join(char_map.keys()), lambda x: char_map[x.group()]).astype(float)

features = features.astype(np.float32)
features = features.values.reshape(features.shape[0], -1)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=32, activation='relu', input_shape=(features.shape[1],1)),
    tf.keras.layers.Dense(units=4, activation='softmax')
])

# Compile the model with a categorical crossentropy loss function and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Convert the labels to one-hot encoding
labels_one_hot = pd.get_dummies(labels)

# Train the model on the features and one-hot encoded labels
history = model.fit(features, labels_one_hot.values, epochs=10)

# Evaluate the model on the training set
y_pred = np.argmax(model.predict(features), axis=1)
y_true = np.argmax(labels_one_hot.values, axis=1)

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print("Training Accuracy: {:.2f}%".format(accuracy * 100))
print("Training F1 Score: {:.2f}".format(f1))