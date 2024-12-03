
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def parse_embedding(embedding_str):
    """
    Parse the embedding string into a list of floats.
    """
    try:
        return list(map(float, embedding_str.strip('[]').split()))
    except (ValueError, AttributeError):
        return None

def load_and_process_data(medcat_path, spacy_path):
    """
    Load MedCAT and SpaCy embeddings, parse them, and identify common attributes.
    """
    # Load datasets
    structured_medcat_embeddings = pd.read_csv(medcat_path)
    structured_spacy_embeddings = pd.read_csv(spacy_path)
    
    # Parse embeddings
    structured_medcat_embeddings['Parsed_Embeddings'] = structured_medcat_embeddings['UMLS_Embeddings'].apply(parse_embedding)
    structured_spacy_embeddings['Parsed_Embeddings'] = structured_spacy_embeddings['UMLS_Embeddings'].apply(parse_embedding)
    
    # Identify common attributes
    structured_attributes = list(
        set(structured_medcat_embeddings['Attribute']).intersection(set(structured_spacy_embeddings['Attribute']))
    )
    print(f"Found {len(structured_attributes)} common attributes.")
    
    return structured_medcat_embeddings, structured_spacy_embeddings, structured_attributes

def filter_attributes(structured_attributes, keywords_to_exclude):
    """
    Filter out attributes containing specific keywords.
    """
    return [
        attr for attr in structured_attributes
        if not any(keyword in attr.lower() for keyword in keywords_to_exclude)
    ]

def calculate_cosine_similarity(structured_attributes, medcat_embeddings, spacy_embeddings):
    """
    Calculate cosine similarity between MedCAT and SpaCy embeddings.
    """
    results = []
    for attribute in structured_attributes:
        # Get embeddings
        medcat_row = medcat_embeddings[medcat_embeddings['Attribute'] == attribute]
        spacy_row = spacy_embeddings[spacy_embeddings['Attribute'] == attribute]
        
        medcat_embedding = medcat_row['Parsed_Embeddings'].values[0] if not medcat_row.empty else None
        spacy_embedding = spacy_row['Parsed_Embeddings'].values[0] if not spacy_row.empty else None
        
        if medcat_embedding is not None and spacy_embedding is not None:
            similarity = cosine_similarity([medcat_embedding], [spacy_embedding])[0, 0]
            medcat_pretty_name = medcat_row['pretty_name'].values[0]
            spacy_pretty_name = spacy_row['pretty_name'].values[0]
            
            results.append({
                "Attribute": attribute,
                "MedCAT Pretty Name": medcat_pretty_name,
                "SpaCy Pretty Name": spacy_pretty_name,
                "Cosine Similarity": similarity
            })
    return pd.DataFrame(results)

def perform_clustering(combined_embeddings, n_clusters=5):
    """
    Perform KMeans clustering on combined embeddings and visualize results.
    """
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(combined_embeddings)
    labels = kmeans.labels_
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(combined_embeddings, labels)
    print(f"Silhouette Score: {silhouette_avg}")
    
    # PCA for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)
    
    # Plot clusters
    plt.figure(figsize=(10, 7))
    for cluster_idx in range(n_clusters):
        cluster_points = reduced_embeddings[labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_idx}')
    
    cluster_centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=200, label='Centroids')
    plt.title("KMeans Clustering of Combined Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid()
    plt.show()
    
    return labels

def main():
    # File paths
    medcat_path = "input/structured_keywords/structured_med_embeddings_freq.csv"
    spacy_path = "input/structured_keywords/structured_general_embeddings.csv"
    
    # Keywords to exclude
    keywords_to_exclude = ["pathology", "cancer", "clinical", "icd-o-3", "AJCC"]
    
    # Load and process data
    medcat_embeddings, spacy_embeddings, structured_attributes = load_and_process_data(medcat_path, spacy_path)
    
    # Filter attributes
    structured_attributes = filter_attributes(structured_attributes, keywords_to_exclude)
    
    # Calculate cosine similarity
    similarity_df = calculate_cosine_similarity(structured_attributes, medcat_embeddings, spacy_embeddings)
    similarity_df = similarity_df.sort_values(by='Cosine Similarity', ascending=True)
    similarity_df.to_csv("structured_attribute_similarity.csv", index=False)
    print("Cosine similarity results saved to 'structured_attribute_similarity.csv'.")
    
    # Combine embeddings
    combined_embeddings = []
    filtered_attributes = []
    for attribute in structured_attributes:
        medcat_row = medcat_embeddings[medcat_embeddings['Attribute'] == attribute]
        spacy_row = spacy_embeddings[spacy_embeddings['Attribute'] == attribute]
        
        medcat_embedding = medcat_row['Parsed_Embeddings'].values[0] if not medcat_row.empty else None
        spacy_embedding = spacy_row['Parsed_Embeddings'].values[0] if not spacy_row.empty else None
        
        if medcat_embedding is not None and spacy_embedding is not None:
            combined_embeddings.append(np.concatenate([medcat_embedding, spacy_embedding]))
            filtered_attributes.append(attribute)
    
    combined_embeddings = np.array(combined_embeddings)
    
    # Perform clustering
    labels = perform_clustering(combined_embeddings, n_clusters=5)
    
    # Save clustering results
    clustering_results = pd.DataFrame({
        "Attribute": filtered_attributes,
        "Cluster": labels
    })
    clustering_results.to_csv("kmeans_clustering_results.csv", index=False)
    print("KMeans clustering completed and results saved to 'kmeans_clustering_results.csv'.")

if __name__ == "__main__":
    main()
