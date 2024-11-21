from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

UmlsBERT_path = "path/to/umlsbert/model"
input_csv = "med_name_freq_raw.csv" # path to the csv with terms and their frequencies
output_csv = "med_embeddings_freq.csv"

tokenizer = AutoTokenizer.from_pretrained(UmlsBERT_path)
model = AutoModel.from_pretrained(UmlsBERT_path)

df = pd.read_csv(input_csv)

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze().numpy()

# Add embeddings to the DataFrame
embeddings = []
for term in df['pretty_name']: #TODO
    embedding = get_embedding(term, tokenizer, model)
    embeddings.append(embedding)

df['UMLS_Embeddings'] = embeddings
df.to_csv(output_csv, index=False)