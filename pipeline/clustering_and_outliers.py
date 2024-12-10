import re
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import numpy as np


input_csv = 'General_NER/analysis/input/report_keywords/report_both_embeddings.csv'
grouped_output_csv = 'data/med_embeddings_grouped.csv'  # Output file with clustering results
outliers_output_csv = 'data/med_embeddings_outliers.csv'  # Output file with outlier detection results

def parse_embedding(embedding_str):
    """
    Parses the embedding string from the CSV into a numpy array.
    """
    embedding_str = embedding_str.strip()[1:-1].strip()  # Remove the outer brackets and extra spaces
    embedding_list = list(map(float, embedding_str.split()))  # Split by spaces and convert to float
    return np.array(embedding_list)

# Grouping Keywords by Clinical Semantics
def group_keywords_by_semantics(input_csv, output_csv, eps=0.5, min_samples=1):
    data = pd.read_csv(input_csv)
    embeddings = np.array([parse_embedding(e) for e in data['UMLS_Embeddings']])
    
    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(embeddings)

    data['cluster_id'] = cluster_labels
    data.to_csv(output_csv, index=False)
    print(f"Grouping complete. Results saved to {output_csv}")

# Semantic Outlier Detection
def detect_semantic_outliers(input_csv, output_csv, n_neighbors=20):
    data = pd.read_csv(input_csv)
    embeddings = np.array([parse_embedding(e) for e in data['UMLS_Embeddings']]) 

    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    outlier_scores = lof.fit_predict(embeddings)
    
    # Add outlier labels to the dataframe
    data['outlier'] = (outlier_scores == -1).astype(int)  # 1 for outlier, 0 for inlier
    data.to_csv(output_csv, index=False)
    print(f"Outlier detection complete. Results saved to {output_csv}")


if __name__ == "__main__":
    group_keywords_by_semantics(input_csv, grouped_output_csv)
    detect_semantic_outliers(grouped_output_csv, outliers_output_csv)
