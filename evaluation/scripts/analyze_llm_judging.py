import json
import os
import pandas as pd
from statistics import mean

# Pfade anpassen
INPUT_PATH = "data/processed/full_sampled_with_judging.json"
OUTPUT_DIR = "results/metrics"
OUTPUT_OVERALL_CSV = f"{OUTPUT_DIR}/overall_scores.csv"
OUTPUT_OVERALL_MD = f"{OUTPUT_DIR}/overall_scores.md"
OUTPUT_SECTION_CSV = f"{OUTPUT_DIR}/section_scores.csv"
OUTPUT_SECTION_MD = f"{OUTPUT_DIR}/section_scores.md"

# Modellvarianten und ihre Judging-Felder
variant_fields = {
    "Baseline_no_context": "judging_generated_baseline_no_context",
    "Baseline_with_context": "judging_generated_baseline",
    "Finetuned_no_context": "judging_generated_finetuned_no_context",
    "Finetuned_with_context": "judging_generated_finetuned"
}

# Kriterienliste
criteria = [
    "visibility_principle",
    "context_relevance",
    "entity_naming",
    "informativeness",
    "redundancy_avoidance",
    "style_readability",
    "total"
]

# 1. Daten laden (JSONL oder Array)
data = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    first = f.read(1)
    f.seek(0)
    if first == '[':
        data = json.load(f)
    else:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

# 2. Analyse pro Modellvariante
def calculate_model_means(data, variant_fields, criteria):
    results = []
    for label, field in variant_fields.items():
        entry_metrics = {crit: [] for crit in criteria}
        for entry in data:
            judge = entry.get(field)
            if judge:
                for crit in criteria:
                    val = judge.get(crit)
                    if isinstance(val, (int, float)):
                        entry_metrics[crit].append(val)
        mean_metrics = {crit: mean(entry_metrics[crit]) if entry_metrics[crit] else None for crit in criteria}
        mean_metrics["model_variant"] = label
        results.append(mean_metrics)
    return pd.DataFrame(results).set_index("model_variant")

# 3. Analyse pro Section
def calculate_section_means(data, variant_fields, criteria):
    section_results = []
    for label, field in variant_fields.items():
        section_metrics = {}
        for entry in data:
            section = entry.get("section", "Unknown")
            judge = entry.get(field)
            if judge:
                if section not in section_metrics:
                    section_metrics[section] = {crit: [] for crit in criteria}
                for crit in criteria:
                    val = judge.get(crit)
                    if isinstance(val, (int, float)):
                        section_metrics[section][crit].append(val)
        
        for section, metrics in section_metrics.items():
            mean_metrics = {crit: mean(vals) if vals else None for crit, vals in metrics.items()}
            mean_metrics["section"] = section
            mean_metrics["model_variant"] = label
            section_results.append(mean_metrics)
    
    return pd.DataFrame(section_results)

# Calculate results
df_overall = calculate_model_means(data, variant_fields, criteria)

# Restructure section analysis to create proper MultiIndex
section_data = []
for entry in data:
    section = entry.get("section", "Unknown")
    for model_name, field in variant_fields.items():
        judge = entry.get(field)
        if judge:
            row = {"section": section, "model": model_name}
            for crit in criteria:
                row[crit] = judge.get(crit)
            section_data.append(row)

# Create DataFrame with proper structure
df_sections = pd.DataFrame(section_data)

# Create pivot table with proper MultiIndex
df_sections_pivot = df_sections.pivot_table(
    index="section",
    columns="model",
    values=criteria,
    aggfunc="mean"
)

# Reorder levels to have metrics as first level
df_sections_pivot = df_sections_pivot.reorder_levels([1,0], axis=1)
df_sections_pivot = df_sections_pivot.sort_index(axis=1)

# Sort index and columns for better readability
df_sections_pivot.sort_index(inplace=True)
df_sections_pivot.columns = pd.MultiIndex.from_tuples(df_sections_pivot.columns)
df_sections_pivot = df_sections_pivot.reindex(sorted(df_sections_pivot.columns), axis=1)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save results with better formatting
df_overall.to_csv(OUTPUT_OVERALL_CSV)
df_sections_pivot.to_csv(OUTPUT_SECTION_CSV)

# Print results
print("\n## Mean Scores per Metric and Model Variant")
print(df_overall.to_markdown())

print("\n## Mean Scores per Section and Model Variant")
print(df_sections_pivot.to_markdown())

print(f"\nResults saved to:")
print(f"- Overall scores: {OUTPUT_OVERALL_CSV}")
print(f"- Section scores: {OUTPUT_SECTION_CSV}")
