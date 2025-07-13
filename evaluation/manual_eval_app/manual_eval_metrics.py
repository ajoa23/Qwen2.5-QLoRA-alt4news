import sqlite3
import pandas as pd
import os
from pathlib import Path

# --- KONFIGURATION ---
BASE = Path(__file__).parent
# Passe hier an, wo deine DB liegt:
DB_PATH = BASE / "eval_detailed.db"
# Und wo deine CSV mit "section" steht:
CSV_PATH = BASE / "manual_eval_pairwise.csv"
OUTPUT_DIR = BASE / "results" / "manual_metrics"

# Verzeichnisse anlegen
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1) DB öffnen und manuelle Scores laden ---
conn = sqlite3.connect(DB_PATH)
df_db = pd.read_sql_query("SELECT image_id, model_variant, visibility_principle, context_relevance, entity_naming, informativeness, redundancy_avoidance, style_readability, total FROM detailed_evals;", conn)

# --- 2) CSV laden, um section zu bekommen ---
df_csv = pd.read_csv(CSV_PATH, usecols=["image_id", "model_variant", "section"])
# Falls du doppelte Kombinationen hast, entferne Duplikate
df_csv = df_csv.drop_duplicates(subset=["image_id", "model_variant"])

# --- 3) Merge DB + CSV nach section ---
df = pd.merge(df_db, df_csv, on=["image_id", "model_variant"], how="left")
if df["section"].isnull().any():
    missing = df[df["section"].isnull()][["image_id", "model_variant"]].drop_duplicates()
    raise KeyError(f"Für diese Kombinationen fehlt die Sektion in der CSV:\n{missing}")

# --- 4) Kriterien definieren ---
criteria = [
    "visibility_principle",
    "context_relevance",
    "entity_naming",
    "informativeness",
    "redundancy_avoidance",
    "style_readability",
    "total"
]

# --- 5) Overall-Mittelwerte pro Modellvariante ---
df_overall = df.groupby("model_variant")[criteria].mean().reset_index()

# --- 6) Mittelwerte pro Section und Modellvariante ---
df_section = (
    df
    .groupby(["section", "model_variant"])[criteria]
    .mean()
    .reset_index()
    .pivot(index="section", columns="model_variant", values=criteria)
)

# --- 7) Ausgabe in Konsole ---
print("## Overall Mean Scores per Model Variant\n")
print(df_overall.to_markdown(index=False))
print("\n## Mean Scores per Section & Model Variant\n")
print(df_section.to_markdown())

# --- 8) Speichere CSV & Markdown ---
(df_overall
    .to_csv(OUTPUT_DIR / "manual_overall_scores.csv", index=False))
with open(OUTPUT_DIR / "manual_overall_scores.md", "w", encoding="utf-8") as f:
    f.write("# Manual Evaluation – Mean Scores per Model Variant\n\n")
    f.write(df_overall.to_markdown(index=False))

df_section.to_csv(OUTPUT_DIR / "manual_section_scores.csv")
with open(OUTPUT_DIR / "manual_section_scores.md", "w", encoding="utf-8") as f:
    f.write("# Manual Evaluation – Mean Scores per Section & Model Variant\n\n")
    f.write(df_section.to_markdown())

print(f"\nErgebnisse gespeichert in: {OUTPUT_DIR}")
