import pandas as pd
import spacy
from tqdm import tqdm
import re

nlp = spacy.load("en_core_web_sm")
input_file = "data/NLST_concatenated.csv"
output_file = "output/structured_general_term.csv"
df = pd.read_csv(input_file)

required_columns = ["Label", "Description", "Concatenated"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Input file must contain the following columns: {required_columns}")
df["NER_Text"] = df["Description"].fillna("") + " " + df["Concatenated"].fillna("")

def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        # print('ent')
        if not re.search(r'\d', ent.text):
            entities.append({"Pretty Name": ent.text.strip(), "Type": ent.label_})
    return entities

results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing NER"):
    label = row["Label"]
    ner_text = row["NER_Text"]
    entities = extract_entities(ner_text)
    for entity in entities:
        results.append({
            "Attribute": label,
            "Pretty Name": entity["Pretty Name"],
            "Entity Type": entity["Type"],
        })

results_df = pd.DataFrame(results)
results_df = results_df.drop_duplicates()
results_df.to_csv(output_file, index=False)

print(f"NER results saved to {output_file}")
print(results_df.head(10))
