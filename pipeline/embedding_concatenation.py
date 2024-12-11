import pandas as pd
import numpy as np

report_files = ["General_NER/analysis/input/report_keywords/report_med_embeddings_freq.csv", "General_NER/analysis/input/report_keywords/report_general_embeddings_freq.csv"]
structured_files = ["General_NER/analysis/input/structured_keywords/structured_med_embeddings_freq.csv", "General_NER/analysis/input/structured_keywords/structured_general_embeddings.csv"]

# Load and concatenate files
def load_and_concatenate(files):
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        df["UMLS_Embeddings"] = df["UMLS_Embeddings"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" ")
        )  # Parse embeddings
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Concatenate files
report_embeddings = load_and_concatenate(report_files)
structured_embeddings = load_and_concatenate(structured_files)

report_embeddings.to_csv("General_NER/analysis/input/report_keywords/report_both_embeddings.csv")
structured_embeddings.to_csv("General_NER/analysis/input/report_keywords/structured_both_embeddings.csv")
