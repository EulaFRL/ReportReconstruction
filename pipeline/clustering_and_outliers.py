import re
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import numpy as np


input_csv = 'data/report_keywords/report_both_embeddings.csv'
grouped_output_csv = 'data/report_keywords/report_embeddings_grouped.csv'  # Output file with clustering results
outliers_output_csv = 'data/report_keywords/raw_report_embeddings_outliers.csv'  # Output file with outlier detection results
top_outliers_csv = 'data/report_keywords/top20_report_embeddings_outliers.csv'

def parse_embedding(embedding_str):
    """
    Parses the embedding string from the CSV into a numpy array.
    """
    embedding_str = embedding_str.strip()[1:-1].strip()  # Remove the outer brackets and extra spaces
    embedding_list = list(map(float, embedding_str.split()))  # Split by spaces and convert to float
    return np.array(embedding_list)

def group_keywords_by_semantics(input_csv, output_csv, eps=0.5, min_samples=1, output_txt="clustered_pretty_names.txt"):
    import pandas as pd
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    # Helper function to parse embeddings
    def parse_embedding(embedding_str):
        return np.fromstring(embedding_str.strip('[]'), sep=',')
    
    # Load data and prepare embeddings
    data = pd.read_csv(input_csv)
    embeddings = np.array([parse_embedding(e) for e in data['UMLS_Embeddings']])
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(embeddings)

    # Assign cluster labels to data
    data['cluster_id'] = cluster_labels
    data.to_csv(output_csv, index=False)
    print(f"Grouping complete. Results saved to {output_csv}")
    
    # # Write 'pretty_name' column for clusters with more than one row to a text file
    # with open(output_txt, 'w') as file:
    #     grouped = data.groupby('cluster_id')
    #     for cluster_id, group in grouped:
    #         if len(group) > 1:  # Check if the cluster has more than one row
    #             file.write(f"Cluster ID {cluster_id}:\n")
    #             file.write(group['pretty_name'].to_string(index=False))
    #             file.write("\n" + "-" * 40 + "\n")  # Separator for better readability
    # print(f"Cluster details with multiple rows saved to {output_txt}")


# Semantic Outlier Detection
def detect_semantic_outliers(input_csv, output_csv, outlier_csv, n_neighbors=20):
    data = pd.read_csv(input_csv)
    embeddings = np.array([parse_embedding(e) for e in data['UMLS_Embeddings']]) 

    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    outlier_scores = lof.fit_predict(embeddings)
    
    # Add outlier labels to the dataframe
    data['outlier'] = (outlier_scores == -1).astype(int)  # 1 for outlier, 0 for inlier
    # Sort the DataFrame by 'frequency' in ascending order
    sorted_data = data.sort_values(by='frequency', ascending=True)

    # Extract the top 20 rows with the lowest frequencies
    top_20_lowest = sorted_data.head(20)
    data.to_csv(outlier_csv, index=False)

    data.to_csv(output_csv, index=False)
    print(f"Outlier detection complete. Results saved to {output_csv}")


if __name__ == "__main__":
    group_keywords_by_semantics(input_csv, grouped_output_csv)
    detect_semantic_outliers(grouped_output_csv, outliers_output_csv, top_outliers_csv)
