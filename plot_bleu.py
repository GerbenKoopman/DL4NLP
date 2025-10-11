import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv("chrf.csv")

# Set the x-axis values
x = df["train/global_step"]

# Columns to plot
columns_to_plot = [
    "be_uk_en",
    "az_tr_en",
    "az_tr_en & be_uk_en-12",
    "az_tr_en baseline",
    "be_uk_en baseline",
    "az_tren & be_uk_en baseline",
]

# Create a new plot
plt.figure(figsize=(12, 8))

# Plot each specified column
for column in columns_to_plot:
    if column in df.columns:
        plt.plot(x, df[column], label=column)

# Add labels and title for clarity
plt.xlabel("Global Step")
plt.ylabel("CHRF Score")
plt.title("CHRF Score vs. Global Step for Different Models")
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig("chrf_plot.png")

print("Plot saved to chrf_plot.png")
