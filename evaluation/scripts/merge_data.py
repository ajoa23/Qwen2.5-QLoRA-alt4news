import json

# Lade beide Dateien erneut nach Kernel-Reset
with open("data/processed/testset_with_predictions_20250602_220302.json", "r") as f1:
    with_context = [json.loads(line) for line in f1]

with open("data/processed/testset_with_predictions_no_context20250603_193545.json", "r") as f2:
    no_context = [json.loads(line) for line in f2]

# Erstelle ein Dictionary mit image_id als Schlüssel für die no_context-Datei
no_context_dict = {entry["image_id"]: entry for entry in no_context}

# Füge die _no_context-Felder hinzu
for entry in with_context:
    image_id = entry["image_id"]
    if image_id in no_context_dict:
        no_entry = no_context_dict[image_id]
        entry["generated_baseline_no_context"] = no_entry.get("generated_baseline", "")
        entry["generated_finetuned_no_context"] = no_entry.get("generated_finetuned", "")

# Speichere die zusammengeführte Datei
output_path = "data/processed/merged_predictions_with_no_context.json"
with open(output_path, "w") as f_out:
    for entry in with_context:
        f_out.write(json.dumps(entry) + "\n")

output_path
