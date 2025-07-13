import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset

# ---- Pfade konfigurieren ----
INPUT_PATH = "data/processed/merged_predictions_with_no_context_final.json"
RESULTS_DIR = "results"

# ---- Spalten, die analysiert werden sollen ----
text_columns_to_analyze = {
    "reference": "openai_alt_text_refined",
    "baseline": "generated_baseline",
    "finetuned": "generated_finetuned",
    "baseline_no_context": "generated_baseline_no_context",
    "finetuned_no_context": "generated_finetuned_no_context"
}

# ---- Lade das Testset ----
print(f"📥 Lade angereichertes Testset: {INPUT_PATH}")
ds_dict = load_dataset("json", data_files=INPUT_PATH)
ds = ds_dict["train"]
print(f"✅ Geladen: {len(ds)} Beispiele\n")

# ---- Analysefunktionen ----

def analyze_alt_text_length_stats(ds, text_column, output_dir, plot_histogram=True, histogram_bins=20):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if text_column not in ds.column_names:
        print(f"⚠️ Spalte '{text_column}' nicht gefunden.")
        return

    df = pd.DataFrame(ds[:])
    df["alt_text_length"] = df[text_column].fillna("").apply(lambda x: len(str(x).strip()))
    lengths = df["alt_text_length"]

    print(f"\n📊 Statistik für Alt-Text-Längen ({text_column}):")
    print(f"  ➤ Mittelwert : {lengths.mean():.2f}")
    print(f"  ➤ Median     : {lengths.median():.2f}")
    print(f"  ➤ Std-Abw    : {lengths.std():.2f}")
    print(f"  ➤ Min        : {lengths.min()}")
    print(f"  ➤ Max        : {lengths.max()}")

    if plot_histogram:
        plt.figure(figsize=(7, 4))
        plt.hist(lengths, bins=histogram_bins, edgecolor="black")
        plt.axvline(150, color="red", linestyle="--", linewidth=1, label="Limit 150")
        plt.xlabel("Alt-Text Länge (Zeichen)")
        plt.ylabel("Häufigkeit")
        plt.title(f"Alt-Text Längenverteilung – {text_column}")
        plt.legend()
        plt.tight_layout()
        plot_file = Path(output_dir) / f"alt_text_length_histogram.png"
        plt.savefig(plot_file, dpi=120)
        plt.close()
        print(f"📈 Histogramm gespeichert unter: {plot_file}")

def analyze_alt_text_lengths(ds, text_column, output_dir, length_threshold=150):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if text_column not in ds.column_names:
        return

    long_ds = ds.filter(
        lambda x: len(str(x[text_column]).strip()) > length_threshold,
        batched=False
    )

    print(f"🔍 {long_ds.num_rows} Alt-Texte über {length_threshold} Zeichen ({text_column})")

    if long_ds.num_rows > 0:
        df = pd.DataFrame(long_ds[:])
        df["alt_text_length"] = df[text_column].apply(lambda x: len(str(x).strip()))
        cols = ["image_id", "headline", "caption", "section", text_column, "alt_text_length"]
        df = df[[c for c in cols if c in df.columns]]
        output_file = Path(output_dir) / f"long_alt_texts.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Gespeichert unter: {output_file}")

def analyze_empty_alt_texts(ds, text_column):
    if text_column not in ds.column_names:
        return

    empty_ds = ds.filter(
        lambda x: str(x[text_column]).strip() == "" or x[text_column] is None,
        batched=False
    )

    print(f"🟡 {empty_ds.num_rows} leere Alt-Texte ({text_column})")

def analyze_picture_image_starts(ds, text_column, output_dir, patterns=None):
    if patterns is None:
        patterns = (
            "picture of", "image of", "this image", "this picture", 
            "this is a picture of", "this is an image of"
        )

    pattern_regex = r"^\s*(" + "|".join(patterns) + r")\s+of"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if text_column not in ds.column_names:
        return

    pi_ds = ds.filter(
        lambda x: x[text_column] is not None and re.match(pattern_regex, str(x[text_column]).strip(), re.IGNORECASE),
        batched=False
    )

    print(f"🔍 {pi_ds.num_rows} Alt-Texte beginnen mit Bild-bezogenen Phrasen ({text_column})")

    if pi_ds.num_rows > 0:
        df = pd.DataFrame(pi_ds[:])
        cols = ["image_id", "section", text_column]
        df = df[[c for c in cols if c in df.columns]]
        output_file = Path(output_dir) / f"alt_text_starts_with_picture_image.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Gespeichert unter: {output_file}")

# ---- Hauptausführung ----

# Übersicht für alle Metriken sammeln
metrics_summary = []

for label, column_name in text_columns_to_analyze.items():
    print(f"\n🔍 Analyse für: {label.upper()} ({column_name})\n{'-'*50}")
    output_path = f"{RESULTS_DIR}/merged/{label}"

    # Längenstatistik berechnen und speichern
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if column_name in ds.column_names:
        df = pd.DataFrame(ds[:])
        df["alt_text_length"] = df[column_name].fillna("").apply(lambda x: len(str(x).strip()))
        lengths = df["alt_text_length"]
        metrics_summary.append({
            "variant": label,
            "mean_length": lengths.mean(),
            "median_length": lengths.median(),
            "std_length": lengths.std(),
            "min_length": lengths.min(),
            "max_length": lengths.max(),
            "num_empty": sum(df[column_name].fillna("").apply(lambda x: str(x).strip() == "")),
            "num_long": sum(lengths > 150)
        })

    analyze_alt_text_length_stats(ds, text_column=column_name, output_dir=output_path)
    analyze_empty_alt_texts(ds, text_column=column_name)
    analyze_alt_text_lengths(ds, text_column=column_name, output_dir=output_path)
    analyze_picture_image_starts(ds, text_column=column_name, output_dir=output_path)
    print(f"✅ Analyse abgeschlossen für: {label.upper()}\n{'='*50}\n")

# Export als CSV
if metrics_summary:
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(f"{RESULTS_DIR}/alttext_metrics_overview.csv", index=False)
    print(f"📄 Übersicht über alle Metriken gespeichert unter: {RESULTS_DIR}/alttext_metrics_overview.csv")

print("🔚 Alle Analysen abgeschlossen.")