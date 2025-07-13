import json
import random
from collections import defaultdict
import os

def sample_by_category_with_clean_image_url(
    input_json_path,
    output_json_path,
    category_key="section",
    max_per_category=400,
    seed=42
):
    """
    Führt ein kategoriebasiertes Sampling durch und ergänzt jedes Objekt
    um eine bereinigte Bild-URL unter dem Schlüssel 'image_url_clean'.

    Funktion:
    - Gruppiert Einträge nach `category_key`.
    - Zieht pro Kategorie maximal `max_per_category` Einträge per Zufallsauswahl.
    - Belässt alle Originalfelder unverändert.
    - Fügt ein zusätzliches Feld 'image_url_clean' hinzu (Bild-URL ohne GET-Parameter).
    - Speichert das Ergebnis unter `output_json_path`.

    Parameter:
    - input_json_path (str): Pfad zur Eingabedatei.
    - output_json_path (str): Pfad zur Zieldatei.
    - category_key (str): Kategorisierungsfeld (z. B. "section").
    - max_per_category (int): Maximale Anzahl pro Kategorie.
    - seed (int): Zufallsinitialisierung zur Reproduzierbarkeit.
    """

    random.seed(seed)

    # Datensatz laden
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Gruppierung nach Kategorie
    categories = defaultdict(list)
    for item in data:
        cat = item.get(category_key, "UNKNOWN")
        categories[cat].append(item)

    # Sampling
    sampled = []

    for cat, items in categories.items():
        selected = items if len(items) <= max_per_category else random.sample(items, max_per_category)

        for s in selected:
            new_entry = dict(s)  # Kopie des Originaleintrags
            raw_url = s.get("image", "")
            base_url = raw_url.split("?", 1)[0]
            new_entry["image_url_clean"] = base_url  # Neues Feld ergänzen
            sampled.append(new_entry)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f_out:
        json.dump(sampled, f_out, ensure_ascii=False, indent=2)

    print(f"[✓] Gesamtanzahl: {len(sampled)} Einträge in {output_json_path} geschrieben.")


# Beispielhafte Ausführung
if __name__ == "__main__":
    sample_by_category_with_clean_image_url(
        input_json_path="data/raw/nytimes.json",
        output_json_path="data/processed/full_sampled_with_image_url_clean.json",
        category_key="section",
        max_per_category=400,
        seed=42
    )
