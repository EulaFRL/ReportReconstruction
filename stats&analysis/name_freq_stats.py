import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load CSV data into a DataFrame
data = pd.read_csv("data/med_name_freq_raw.csv")


# Logarithmic binning for frequencies
frequency_bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
frequency_hist, frequency_edges = np.histogram(data["frequency"], bins=frequency_bins)

# Linear binning for average accuracy
accuracy_bins = np.linspace(0.7, 1.0, 4)  # 0.0 to 1.0 in 0.1 intervals
accuracy_hist, accuracy_edges = np.histogram(data["average_acc"], bins=accuracy_bins)

# Plot frequency distribution and save as image
plt.figure(figsize=(10, 5))
plt.bar(range(len(frequency_hist)), frequency_hist, width=0.8, 
        tick_label=[f"{frequency_bins[i]}-{frequency_bins[i+1]}" for i in range(len(frequency_bins)-1)])
plt.xticks(rotation=45)
plt.title("Distribution of Pretty Name Frequencies")
plt.xlabel("Frequency Ranges")
plt.ylabel("Number of Pretty Names")
plt.tight_layout()
plt.savefig("pretty_name_frequency_distribution.png")
plt.close()

# Plot average accuracy distribution and save as image
plt.figure(figsize=(10, 5))
plt.bar(range(len(accuracy_hist)), accuracy_hist, width=0.8, 
        tick_label=[f"{accuracy_bins[i]:.1f}-{accuracy_bins[i+1]:.1f}" for i in range(len(accuracy_bins)-1)])
plt.xticks(rotation=45)
plt.title("Distribution of Pretty Name Average Confidence")
plt.xlabel("Average Accuracy Ranges")
plt.ylabel("Number of Pretty Names")
plt.tight_layout()
plt.savefig("pretty_name_confidence_distribution.png")
plt.close()
