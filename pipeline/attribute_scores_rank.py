import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def parse_embedding(embedding_str):
    """
    Parses the embedding string from the CSV into a numpy array.
    """
    embedding_str = embedding_str.strip()[1:-1].strip()  # Remove the outer brackets and extra spaces
    embedding_list = list(map(float, embedding_str.split()))  # Split by spaces and convert to float
    return np.array(embedding_list)

def calculate_keyword_coverage(attribute_csv, grouped_keywords_csv, output_csv, similarity_threshold=0.7):
    """
    Calculates the contribution of each structured attribute to the coverage of report keywords.
    Args:
        attribute_csv (str): Path to the CSV file with attributes and embeddings.
        grouped_keywords_csv (str): Path to the CSV file with grouped keywords and embeddings.
        output_csv (str): Path to save the output file with coverage scores.
        similarity_threshold (float): Minimum cosine similarity to consider a keyword covered by an attribute.
    """
    # Load attributes and grouped keywords
    attributes = pd.read_csv(attribute_csv)
    keywords = pd.read_csv(grouped_keywords_csv)
    
    # Parse embeddings
    attribute_embeddings = np.array([parse_embedding(attr) for attr in attributes['UMLS_Embeddings']])
    keyword_embeddings = np.array([parse_embedding(kw) for kw in keywords['UMLS_Embeddings']])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(attribute_embeddings, keyword_embeddings)
    
    # Calculate the maximum similarity for each attribute
    max_similarity_scores = similarity_matrix.max(axis=1)
    attributes['max_similarity'] = max_similarity_scores

    # Calculate coverage for each attribute
    coverage_scores = (similarity_matrix > similarity_threshold).sum(axis=1)
    attributes['coverage_score'] = coverage_scores
    
    # Save the results
    attributes.to_csv(output_csv, index=False)
    print(f"Coverage calculation complete. Results saved to {output_csv}")


def calculate_composite_score(attribute_csv, outlier_csv, output_csv, 
                              alpha=0.5, beta=0.3, gamma=0.2, similarity_column='max_similarity', 
                              coverage_column='coverage_score', outlier_column='outlier'):
    """
    Calculates a composite score for each attribute and ranks them.
    Args:
        attribute_csv (str): Path to the CSV file with attribute embeddings and coverage scores.
        outlier_csv (str): Path to the CSV file with outlier information for attributes.
        output_csv (str): Path to save the output file with composite scores and rankings.
        alpha, beta, gamma (float): Weights for similarity, coverage, and outlier alignment, respectively.
        similarity_column (str): Column name for similarity scores in the attribute CSV.
        coverage_column (str): Column name for coverage scores in the attribute CSV.
        outlier_column (str): Column name for outlier alignment in the outlier CSV.
    """
    # Load attribute and outlier data
    attributes = pd.read_csv(attribute_csv)
    outliers = pd.read_csv(outlier_csv)
    
    # Merge attributes with outlier flags
    merged = attributes.merge(outliers[['pretty_name', outlier_column]], on='pretty_name', how='left')
    merged[outlier_column] = merged[outlier_column].fillna(0)  # Fill missing outlier flags with 0
    
    # Calculate composite score
    merged['composite_score'] = (
        alpha * merged[similarity_column] +
        beta * merged[coverage_column] +
        gamma * merged[outlier_column]
    )
    
    # Rank attributes by composite score
    merged['rank'] = merged['composite_score'].rank(ascending=False)
    
    # Save results
    merged.to_csv(output_csv, index=False)
    print(f"Composite scoring complete. Results saved to {output_csv}")

if __name__ == '__main__':
    attribute_csv = 'General_NER/analysis/input/structured_keywords/structured_both_embeddings.csv'
    grouped_keywords_csv = '../data/report_keywords/med_embeddings_grouped.csv'
    outlier_csv = '../data/report_keywords/med_embeddings_outliers.csv'
    coverage_output_csv = '../data/attributes_with_coverage.csv'
    final_output_csv = '../data/final_ranked_attributes.csv'

    # Step 1: Calculate keyword coverage
    calculate_keyword_coverage(attribute_csv, grouped_keywords_csv, coverage_output_csv)

    # TODO: parameters
    # Step 2: Calculate composite scores and rank attributes
    calculate_composite_score(coverage_output_csv, outlier_csv, final_output_csv,
                            alpha=0.5, beta=0.3, gamma=0.2)
