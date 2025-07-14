# Manual Evaluation App

Dieses Tool dient der manuellen Bewertung von Alt-Texten für Bilder aus dem N24News-Datensatz.  
Verglichen werden jeweils die Baseline- und die Fine-Tuned-Variante auf sechs Qualitätsdimensionen und Gesamteindruck.

## Setup

1. Python 3.10+ installieren
2. Abhängigkeiten installieren:
   ```sh
   pip install -r requirements.txt
   ```
3. Die Datei `manual_eval_pairwise.csv` muss im App-Ordner liegen (siehe Datenaufbereitung).

## Nutzung

Starte die App mit:
```sh
streamlit run app.py
```
Für Einzelbewertungen kann alternativ `app_detail.py` verwendet werden.

## Bewertung

- Für jedes Bild werden Headline, Abstract und Caption angezeigt.
- Beide Alt-Text-Varianten können direkt nebeneinander bewertet werden.
- Die Bewertungen werden in einer SQLite-Datenbank (`eval_detailed.db`) gespeichert.

## Export der Ergebnisse

Aggregierte Scores können mit folgendem Skript berechnet und als Markdown/CSV exportiert werden:
```sh
python manual_eval_metrics.py
```
Die Ergebnisse finden sich im Ordner `results/manual_metrics/`.

## Hinweise

- Die App ist für interne, wissenschaftliche Zwecke konzipiert.
- Für Fragen oder Feedback bitte an den Projektverantwortlichen wenden.
