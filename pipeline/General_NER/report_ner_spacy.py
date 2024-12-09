import pandas as pd
from collections import Counter
import spacy
nlp = spacy.load("en_core_web_sm")

file_path = "data/lung_cancer_notes.csv"
df = pd.read_csv(file_path)
if "TEXT" in df.columns and "DESCRIPTION" in df.columns:
    df["combined_text"] = df["DESCRIPTION"].fillna("") + " " + df["TEXT"].fillna("")
else:
    raise ValueError("The CSV file must contain 'TEXT' and 'DESCRIPTION' columns.")

def extract_attributes(text):
    doc = nlp(text)
    attributes = []
    for chunk in doc.noun_chunks: 
        attributes.append(chunk.text.lower())
    for token in doc:  
        if not token.is_stop and not token.is_punct:
            attributes.append(token.text.lower())
    return attributes


attribute_counter = Counter()
for combined_text in df["combined_text"]:
    attributes = extract_attributes(combined_text)
    attribute_counter.update(attributes)


attribute_df = pd.DataFrame(attribute_counter.items(), columns=["Attribute", "Frequency"]).sort_values(
    by="Frequency", ascending=False
)
print(attribute_df.head(20))

output_file = "output/report_attr_freq.csv"
attribute_df.to_csv(output_file, index=False)
print(f"Attribute frequencies saved to {output_file}")
