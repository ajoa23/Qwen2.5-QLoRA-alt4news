import json
import pandas as pd
import random

# Inputdatei laden
with open("data/processed/testset_with_predictions_20250602_220302.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]  # falls NDJSON

# Optional: als DataFrame für einfaches Handling
df = pd.DataFrame(data)

# Verfügbare Sektionen anzeigen
print("Verfügbare Sektionen:", df["section"].unique())

# Ziel: 96 Beispiele → 4 pro Kategorie, falls 24 Sektionen
samples_per_section = 4
selected_entries = []

for section in df["section"].unique():
    section_df = df[df["section"] == section]
    sampled = section_df.sample(n=samples_per_section, random_state=42)
    selected_entries.extend(sampled.to_dict(orient="records"))

# Als CSV speichern für Streamlit
output_df = pd.DataFrame(selected_entries)

# Optional: Auswahl der Spalten
output_df = output_df[[
    "image_id",
    "image_url_clean",
    "headline",
    "abstract",
    "caption",
    "openai_alt_text_refined",
    "generated_baseline",
    "generated_finetuned",
    "section"
]]

output_df.to_csv("manual_eval_app/manual_eval_sample.csv", index=False)
print("✅ 96 Beispiele gespeichert in manual_eval_sample.csv")

# Aus einem Beispiel zwei machen: 1x baseline, 1x fine-tuned
baseline_df = output_df.copy()
baseline_df["model_variant"] = "baseline"
baseline_df["alt_text"] = baseline_df["generated_baseline"]

finetuned_df = output_df.copy()
finetuned_df["model_variant"] = "finetuned"
finetuned_df["alt_text"] = finetuned_df["generated_finetuned"]

# Kombinieren und sortieren (damit pro Bild Baseline & Fine-Tuned direkt beieinander stehen)
final_df = pd.concat([baseline_df, finetuned_df])
final_df = final_df.sort_values(by=["image_id", "model_variant"])

# Unnötige Spalten entfernen
final_df = final_df[[
    "image_id",
    "image_url_clean",
    "headline",
    "abstract",
    "caption",
    "alt_text",
    "model_variant",
    "section"
]]

# Exportieren
final_df.to_csv("manual_eval_app/manual_eval_pairwise.csv", index=False)
print("✅ Paarweise Vergleichsdatei gespeichert als manual_eval_pairwise.csv")