#!pip install transformers[sentencepiece]
from transformers import pipeline
import pandas as pd
import numpy as np  
from tqdm import tqdm
    
# dummy
# def classification():
#     classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
#     sequence_to_classify = "big alligator eating a banana"
#     candidate_labels = ["medical", 'non-medical']
#     output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
#     print(output)

def classify(text, classifier):
    candidate_labels = ['medical', 'non-medical']
    output = classifier(text, candidate_labels, multi_label=False)
    accuracy = max(output['scores'])
    label = output['labels'][0]
    
    return (accuracy, label)

def main():
    df = pd.read_csv('output/report_ranked_cosine_similarity.csv')
    # columns: ['General Attribute', 'General Frequency', 'Matched Medical Attributes',
    # 'Matched Medical Frequencies', 'Mean Cosine Similarity', 'Direction']
    
    # attr for feeding: 'General Attribute', 'Matched Medical Attributes'
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    
    results = []
    
    print(len(df))

    for _, row in tqdm(df.iterrows(), desc="processing rows"):
        text_to_classify = f"{row['General Attribute']} {row['Matched Medical Attributes']}"
        print(f"Classifying: {text_to_classify}")
        accuracy, label = classify(text_to_classify, classifier)
        print(f"Label is {label}, Accuracy is: {accuracy} \n")
        results.append({
            "General Attribute": row["General Attribute"],
            "Matched Medical Attributes": row["Matched Medical Attributes"],
            "Accuracy": accuracy,
            "Label": label
        })

    result_df = pd.DataFrame(results)

    medical_df = result_df[result_df["Label"] == "medical"].copy()
    general_df = result_df[result_df["Label"] == "non-medical"].copy()

    medical_df = medical_df.sort_values(by="Accuracy", ascending=True) 
    general_df = general_df.sort_values(by="Accuracy", ascending=False)


    medical_df.to_csv('output/report_medical_classification.csv', index=False)
    general_df.to_csv('output/report_general_classification.csv', index=False)



    
if __name__ == '__main__':
    main()