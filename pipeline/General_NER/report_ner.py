import pandas as pd
import spacy
from tqdm import tqdm
import re
from collections import Counter

nlp = spacy.load("en_core_web_sm")

input_file = "lung_cancer_notes.csv"
output_file = "report_ner_results.csv"

df = pd.read_csv(input_file)

if "TEXT" not in df.columns:
    raise ValueError(f"Input file must contain the 'TEXT' column.")

def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if not re.search(r'\d', ent.text):
            entities.append((ent.text.strip(), ent.label_))
    return entities

entity_counter = Counter()

all_entities = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing NER"):
    text = row["TEXT"]
    entities = extract_entities(text)
    all_entities.extend(entities)  
    entity_counter.update(entities)  

results = [
    {"Pretty Name": entity[0], "Frequency": frequency, "Entity Type": entity[1]}
    for entity, frequency in entity_counter.items()
]

results_df = pd.DataFrame(results)
results_df = results_df.drop_duplicates()
results_df.to_csv(output_file, index=False)

print(f"NER results saved to {output_file}")
print(results_df.head(10))
