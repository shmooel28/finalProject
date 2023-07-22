import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def create_chart(data):
    # Create a bar chart from the data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(data.keys(), data.values())
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title('User Input Data Chart')
    return fig


def on_submit():
    data = {}
    for entry in data_entries:
        category = entry[0].get()
        value = entry[1].get()

        if category and value:
            try:
                data[category] = float(value)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
                return
        else:
            messagebox.showerror("Missing Input", "Please fill in all fields.")
            return

    # Show a message with the period result:
    result = "xx"
    accuracy = "yy%"
    message = ("Period: " + result + "\n" + accuracy + "\n")

    messagebox.showinfo("testing result ", message)



# Create the main application window
root = tk.Tk()
root.title("User Data Chart")

input_names = ["IRSL_Net", "Error", "IRSL_front", "error", "Depletion", "Error", "OSL_Net", "Error",
               "OSL_front", "error", "Depletion", "Error", "IRSL/OSL", "Error"]

# Create labels and entries for user input
data_entries = []
for i in range(14):
    lbl = tk.Label(root, text=input_names[i])
    lbl.grid(row=i, column=0, padx=5, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    data_entries.append((entry, entry))

# Create a button to submit the data
submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.grid(row=15, column=0, columnspan=2, padx=5, pady=10)

# Start the GUI event loop
root.mainloop()
