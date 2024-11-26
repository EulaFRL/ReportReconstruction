import json
from medcat.cat import CAT
import pandas as pd
import os

# Load the MedCAT model
cat = CAT.load_model_pack('/Users/katiesmc/Downloads/umls_self_train_model_pt2ch_3760d588371755d0.zip')
print('Model loaded successfully')

# Load the dataset
notes = pd.read_csv('lung_cancer_notes.csv')

# Function to process a batch of notes
def process_batch(notes_batch, batch_number):
    batch_entities = []
    for idx, note in notes_batch.iterrows():
        text = note['TEXT']
        entities = cat.get_entities(text)  # Extract entities using MedCAT
        batch_entities.append({"note_id": idx, "entities": entities})  # Include note_id for traceability

    # Output to JSON file
    output_file = f'lung_cancer_entities_batch_{batch_number}.json'
    with open(output_file, 'w') as f:
        json.dump(batch_entities, f, indent=4)
    print(f"Batch {batch_number} saved to {output_file}")

# Parameters for batching
batch_size = 1000  # Number of notes to process per batch
total_notes = len(notes)
batches = (total_notes // batch_size) + (1 if total_notes % batch_size != 0 else 0)  # Total number of batches

# Process the dataset in batches
for batch_number in range(batches):
    start_idx = batch_number * batch_size
    end_idx = min(start_idx + batch_size, total_notes)
    print(f"Processing batch {batch_number + 1}/{batches} (notes {start_idx} to {end_idx - 1})")

    # Select the current batch
    notes_batch = notes.iloc[start_idx:end_idx]

    # Process and save the batch
    process_batch(notes_batch, batch_number + 1)
