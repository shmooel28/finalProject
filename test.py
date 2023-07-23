import tkinter as tk
from tkinter import filedialog
from tkinter import Toplevel, Text, Scrollbar
import joblib
import pandas as pd

# Load SVM, Decision Tree, and KNN models
svm_model = joblib.load("svm_model.sav")
decision_tree_model = joblib.load("decision_tree_model.pkl")
knn_model = joblib.load("knn_model.sav")


# Function to perform classification using the models
def classify_data(data):
    svm_result = svm_model.predict(data)
    decision_tree_result = decision_tree_model.predict(data)
    knn_result = knn_model.predict(data)
    return svm_result, decision_tree_result, knn_result


# Function to map numeric labels to categories
def map_class_to_category(class_label):
    class_map = {
        0: "Top soil",
        1: "EB",
        2: "CH",
        3: "Hamra"
    }
    return class_map.get(class_label, "Unknown")


# Function to find the majority class label
def find_majority(classes):
    class_count = {}
    for c in classes:
        class_count[c] = class_count.get(c, 0) + 1
    majority_class = max(class_count, key=class_count.get)
    return majority_class


# Function to handle input data from the GUI
def handle_input_data():
    input_names = ["IRSL_Net", "Error", "IRSL_front", "error", "Depletion", "Error", "OSL_Net", "Error",
                   "OSL_front", "error", "Depletion", "Error", "IRSL/OSL", "Error"]

    input_data = []
    for entry in input_entries:
        value = float(entry.get())
        input_data.append(value)

    input_data = [input_data]  # Convert to a list of lists (each list representing a row)
    svm_result, decision_tree_result, knn_result = classify_data(input_data)

    svm_class = map_class_to_category(svm_result[0])
    dt_class = map_class_to_category(decision_tree_result[0])
    knn_class = map_class_to_category(knn_result[0])

    majority_class = find_majority([svm_class, dt_class, knn_class])

    result_label.config(text=f"SVM Result: {svm_class}\nDecision Tree Result: {dt_class}\nKNN Result: {knn_class}\n"
                             f"Majority Result: {majority_class}")


# Function to handle file selection and processing
def handle_file_selection():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    input_names = ["IRSLNet", "Error", "IRSL front", "error", "Depletion", "Error.1", "OSL Net", "Error.2",
                   "OSL front", "error.1", "Depletion.1", "Error.3", "IRSL/OSL", "Error.4"]

    if file_path:
        data_frame = pd.read_excel(file_path)
        columns_count = {}
        for i, col in enumerate(data_frame.columns):
            if col in columns_count:
                data_frame.rename(columns={col: f"{col}_{columns_count[col] + 1}"}, inplace=True)
                columns_count[col] += 1
            else:
                columns_count[col] = 0

        results = []
        for i, row in data_frame.iterrows():
            # take only the relevant columns
            data = [row[data_frame.columns.get_loc(input_name)] for input_name in input_names]
            svm_result, decision_tree_result, knn_result = classify_data([data])
            svm_class = map_class_to_category(svm_result[0])
            dt_class = map_class_to_category(decision_tree_result[0])
            knn_class = map_class_to_category(knn_result[0])
            majority_class = find_majority([svm_class, dt_class, knn_class])
            results.append(f"{i + 1}. SVM Result: {svm_class} Decision Tree Result: {dt_class} KNN Result: {knn_class} "
                           f"Majority Result: {majority_class}\n")

        display_results_window(results)


# Function to display the classification results in a new window
def display_results_window(results):
    result_window = Toplevel(root)
    result_window.title("Classification Results")

    # Create a Text widget to display the results
    text_widget = Text(result_window, wrap=tk.WORD, font=("Arial", 12))
    text_widget.pack(expand=True, fill=tk.BOTH)

    # Add the results to the Text widget
    for result in results:
        text_widget.insert(tk.END, result)

    # Add a scrollbar to the Text widget
    scrollbar = Scrollbar(result_window, command=text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.config(yscrollcommand=scrollbar.set)


# Create the main GUI window
root = tk.Tk()
root.title("Data Classification")

# Input fields for user data entry
input_entries = []
for i, input_name in enumerate(["IRSL_Net", "Error", "IRSL_front", "error", "Depletion", "Error",
                                "OSL_Net", "Error", "OSL_front", "error", "Depletion", "Error", "IRSL/OSL", "Error"]):
    label = tk.Label(root, text=input_name)
    label.grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    input_entries.append(entry)

# Result label to display classification output
result_label = tk.Label(root, text="")
result_label.grid(row=len(input_entries), columnspan=2)

# Buttons for file selection and user data classification
file_button = tk.Button(root, text="Choose XLSX File", command=handle_file_selection)
file_button.grid(row=len(input_entries) + 1, column=0)

classify_button = tk.Button(root, text="Classify Input Data", command=handle_input_data)
classify_button.grid(row=len(input_entries) + 1, column=1)

# Run the main GUI loop
root.mainloop()
