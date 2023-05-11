import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

age_label2int = {
    "Top soil": 0,
    "EB": 1,
    "CH": 2,
    "Hamra": 2,
}
char_map = {'a': '0', 'b': '1', 'c': '2'}

# Load the CSV file into a Pandas DataFrame
data = pd.read_excel("data_b.xlsx")

# Extract the Period column as the labels (Y)
labels = data["Period"]
for i in range(len(labels)):
    if labels[i] in age_label2int:
        labels[i] = age_label2int[labels[i]]
labels = labels.astype(np.int32)

# Extract the rest of the columns as the features (X)
data = data.drop("Period", axis=1)
data["Depth2"] = data["Depth2"].str.replace('|'.join(char_map.keys()), lambda x: char_map[x.group()])
features = data.drop("sample", axis=1)
features = features.astype(np.float32)

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Reshape the input data for the RNN model
X_train = X_train.astype(np.float32).values.reshape((-1, 1, X_train.shape[1]))
X_val = X_val.astype(np.float32).values.reshape((-1, 1, X_val.shape[1]))
X_test = X_test.astype(np.float32).values.reshape((-1, 1, X_test.shape[1]))

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

# Compile the model with a categorical crossentropy loss function and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert the labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train)
y_val_one_hot = tf.keras.utils.to_categorical(y_val)
y_test_one_hot = tf.keras.utils.to_categorical(y_test)

# Train the model on the training set and validate on the validation set
history = model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(X_val, y_val_one_hot))


# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, verbose=2)

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate F1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred_classes, average='weighted')

# Print the results
print('Test accuracy:', test_acc)
print('F1 score:', f1)