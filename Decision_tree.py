import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Define a dictionary to map age labels to integers, if you want to add more ages, you need to add them here
age_label2int = {
    "Top soil": 0,
    "EB": 1,
    "CH": 2,
    "Hamra": 3,
}


def train(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Initialize the Decision Tree model
    dt_model = DecisionTreeClassifier()

    # Fit the Decision Tree model on the training data
    dt_model.fit(X_train, y_train)

    # Test the Decision Tree model
    y_pred = dt_model.predict(X_test)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Test Accuracy: {:.2f}%".format(accuracy * 100))
    print("Test F1 Score: {:.2f}".format(f1))

    return accuracy


if __name__ == "__main__":
    # Load the Excel file into a Pandas DataFrame - the name of the xlsx file
    data = pd.read_excel("data_b.xlsx")

    # Create a new DataFrame with only the ages that you want to learn, in this case we not load Hamra because we don't
    # have enough data
    filtered_data = data.loc[data["Period"].isin(["EB", "CH", "Top soil"])]

    # Convert the age labels to integers using the dictionary
    labels = filtered_data["Period"].apply(lambda x: age_label2int[x])

    features = filtered_data.drop("Period", axis=1)
    # Extract the rest of the columns as the features (X) and drop unnecessary columns, this is with the assumption you
    # don't have any not necessary columns
    features = features.drop(features.columns[0], axis=1)

    # Convert the data types to float32
    labels = labels.astype(np.float32)
    features = features.astype(np.float32)
    sum_test_acc = 0
    times_run = 20
    # Runs the model 20 times to get reliable results
    for _ in range(times_run):
        sum_test_acc += train(features, labels)
    avg_acc = sum_test_acc / times_run
    print("The average accuracy is: {:.2f}%".format(avg_acc * 100))
    dt_model = DecisionTreeClassifier()

    # Fit the Decision Tree model on all the data
    dt_model.fit(features, labels)

    # Save the trained Decision Tree model
    dt_model_file = "decision_tree_model.pkl"
    joblib.dump(dt_model, dt_model_file)
