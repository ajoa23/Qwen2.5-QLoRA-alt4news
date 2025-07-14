# Qwen2.5-QLoRA-alt4news

Code-Repository für die Masterarbeit:  
**"Effiziente Anpassung von Vision-Language-Modellen für kontextbasierte Alt-Text-Generierung: Ein Beitrag zur Barrierefreiheit im Web"**

## Motivation

Das Projekt untersucht, wie Vision-Language-Modelle effizient für die automatische, kontextbasierte Generierung von Alt-Texten für Nachrichtenbilder angepasst werden können. Ziel ist die Verbesserung der Barrierefreiheit im Web durch hochwertige, relevante Bildbeschreibungen.

## Projektstruktur

```
data-preperation/
  data/                # Rohdaten und augmentierte Datensätze
  results/             # Ergebnisse und Vorschauen
  scripts/             # Python-Skripte zur Datenaufbereitung
  src/                 # Prompt-Vorlagen für synthetische Annotation
evaluation/            # Evaluationsdaten, manuelle Bewertung und Skripte
  manual_eval_app/     # Streamlit-App für manuelle Bewertung
fine-tuning/           # Notebooks für das Modell-Finetuning
README.md              # Diese Datei
requirements.txt       # Zentrale Paketliste
```

## Setup

1. **Python installieren** (empfohlen: Version 3.10+)
2. Abhängigkeiten installieren:
   ```sh
   pip install -r requirements.txt
   ```
3. Optional: OpenAI API Key für Alt-Text-Augmentierung setzen (z.B. in `.env`).

## Rohdatensatz

Der Rohdatensatz ist zu groß, um direkt über GitHub bereitgestellt zu werden.  
Er kann über das folgende externe Repository bezogen werden:  
https://github.com/billywzh717/N24News  
Bitte folgen Sie der dortigen Anleitung zum Download und zur lokalen Ablage der Daten.

Alternativ steht ein synthetisch augmentierter Datensatz direkt auf Hugging Face zur Verfügung:  
https://huggingface.co/datasets/Alex23o4/n24news_sample_synthetic_alttext_reduced

## Workflow

1. **Datenaufbereitung**  
   - Skripte in `data-preperation/scripts/` bereiten die Rohdaten auf, augmentieren Alt-Texte (OpenAI), ziehen Stichproben und generieren Vorschauen.
   - Beispiel:  
     - `enrich_alttext_openai.py`: Alt-Texte generieren und verfeinern.
     - `generate_html_preview.py`: HTML-Vorschau für Alt-Texte.

2. **Modell-Finetuning**  
   - Das Notebook `fine-tuning/colab_training_qwen2.5.ipynb` beschreibt das Training des Qwen2.5-Modells mit QLoRA.
   - Trainingsergebnisse werden im Drive gespeichert.

3. **Evaluation**  
   - Automatische Bewertung: Skripte in `evaluation/scripts/` (z.B. BLEU, LLM-Judging).
   - Manuelle Bewertung: Streamlit-App in `evaluation/manual_eval_app/`  
     - App starten:  
       ```sh
       streamlit run evaluation/manual_eval_app/app.py
       ```
     - Ergebnisse werden in SQLite-DB gespeichert und mit `manual_eval_metrics.py` aggregiert.

4. **Ergebnisse & Visualisierung**  
   - Aggregierte Scores und Beispiel-Alt-Texte finden sich in `data-preperation/results/` und `evaluation/results/`.
   - **HTML-Vorschau der generierten Alt-Texte im Vergleich:**  
     [`evaluation/results/alttext_preview.html`](evaluation/results/alttext_preview.html)  
     → Zeigt Baseline, Fine-Tuned und Referenz-Alt-Texte für alle Kategorien im direkten Vergleich.

   - Trainings- und Testdaten:  
     `data-preperation/results/long_alt_texts_train.csv`  
     `data-preperation/results/long_alt_texts_test.csv`

## Komponenten

- **Datenaufbereitung:** Python-Skripte für Sampling, Augmentierung, Vorschau.
- **Modell:** Qwen2.5-VL, Training mit QLoRA/PEFT.
- **Evaluation:** Automatisch (BLEU, LLM-Judging), manuell (Streamlit-App).
- **Visualisierung:** HTML-Preview, Markdown/CSV-Exports.

## Lizenz

Dieses Projekt dient ausschließlich wissenschaftlichen Zwecken im Rahmen der Masterarbeit.

