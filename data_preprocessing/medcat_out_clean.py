import os
import json
import pandas as pd

raw_dir = 'data/medcat_raw'
out_path = 'data/medcat_cleaned.jsonl'

def read_jsons(dir):
    """joins all batches"""
    filenames = os.listdir(dir)
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    
    large_list = []

    for filename in filenames:
        filepath = os.path.join(dir, filename)
        with open(filepath, 'r') as f:
            large_list.extend(json.load(f))
    
    return large_list

def filter_conf(list):
    """filters out entities with confidence <0.75"""
    new_list = []
    for note in list:
        new_note = note
        print(len(new_note["entities"]["entities"].items()))
        entities_to_remove = [entity_key for entity_key in new_note["entities"]["entities"].keys() 
                              if new_note["entities"]["entities"][entity_key]["acc"] < 0.75]
        for entity_key in entities_to_remove:
            print(f"Deleted entity with acc(confidence): {new_note["entities"]["entities"][entity_key]["acc"]}")
            new_note["entities"]["entities"].pop(entity_key)
        print(len(new_note["entities"]["entities"].items()))
        # flatten the uneccessary layer in dictionary
        new_note["tokens"] = new_note["entities"]["tokens"]
        new_note["entities"] = new_note["entities"]["entities"]
        new_list.append(new_note)
    return new_list

def write_json(new_list, path):
    with open(path, 'w+') as f:
        json.dump(new_list, f, indent=4)
    print(f"Successfully written to {path}")

def filter_conf_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # Filter out rows where 'ACC' is lower than 0.75
    filtered_df = df[df['ACC'] >= 0.75]
    filtered_df.to_csv(output_csv, index=False)

    print(filtered_df.head())

if __name__ == '__main__':
    # raw_list = read_jsons(raw_dir)
    # conf_list = filter_conf(raw_list)
    # write_json(conf_list, out_path)

    filter_conf_csv('data/structured_med_freq_raw.csv', 'data/structured_med_freq_cleaned.csv')