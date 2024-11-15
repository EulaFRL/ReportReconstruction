import json
import csv
from collections import defaultdict

in_path = "data/medcat_cleaned.jsonl"
out_path = "data/med_name_freq_raw.csv"

def count_pretty_name(data):
    """
    count frequency, keep cui and calculate average acc(confidence)
    """
    pretty_name_data = defaultdict(lambda: {"cui": None, "acc_sum": 0, "count": 0})

    for note in data:
        for entity in note.get("entities", {}).values():
            pretty_name = entity["pretty_name"]
            cui = entity["cui"]
            acc = entity["acc"]

            if pretty_name_data[pretty_name]["cui"] and pretty_name_data[pretty_name]["cui"] != cui:
                pretty_name_data[pretty_name]["cui"] += f" {cui}"
                print(f"Warning: {pretty_name} has multiple CUIs")
            pretty_name_data[pretty_name]["cui"] = cui
            pretty_name_data[pretty_name]["acc_sum"] += acc
            pretty_name_data[pretty_name]["count"] += 1

    return pretty_name_data

def write_csv(pretty_name_data, out_path):
    csv_data = [
        {
            "pretty_name": name,
            "cui": info["cui"],
            "frequency": info["count"],
            "average_acc": info["acc_sum"] / info["count"]
        }
        for name, info in pretty_name_data.items()
    ]
    
    with open(out_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["pretty_name", "cui", "frequency", "average_acc"])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Successfully written to {out_path}")

def main():
    
    with open(in_path, 'r') as infile:
        data = json.load(infile)
    
    pretty_name_data = count_pretty_name(data)
    
    write_csv(pretty_name_data, out_path)

if __name__ == "__main__":
    main()
