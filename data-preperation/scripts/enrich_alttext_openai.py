import json
import time
import logging
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv

# --------------------------------------------------------------------
# Setup & Konfiguration
# --------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_PATH = "data/processed/full_sampled_with_image_url_clean.json"
OUTPUT_PATH = "data/processed/full_sampled_with_alttext_augmented.json"
PROMPT_GEN_PATH = "prompts/alt_text_generation_prompt.txt"
PROMPT_REF_PATH = "prompts/alt_text_refinement_prompt.txt"

client = OpenAI(api_key=OPENAI_API_KEY)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/alt_text_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_data():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_alt_text(image_url, headline, abstract, caption, prompt_text):
    messages = [
        {"role": "system", "content": prompt_text},
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": 
                f"Headline: {headline}\n"
                f"Abstract: {abstract}\n"
                f"Caption: {caption}\n\n"
                f"Please describe this image following the guidelines above."
            },
            {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "low"
                }
            }
            ]
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=75
        )
        return response.model_dump()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Generation failed: {str(e)}")
        return None

def refine_alt_text(alt_text, headline, abstract, caption, image_url, prompt_text):
    messages = [
        {"role": "system", "content": prompt_text},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":
                    f"Headline: {headline}\n"
                    f"Abstract: {abstract}\n"
                    f"Caption: {caption}\n\n"
                    f"Initial Alt-Text: {alt_text}\n\n"
                    "Please refine this alt text according to the rules above."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "low"
                    }
                }
            ]
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=75
        )
        return response.model_dump()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Refinement failed: {str(e)}")
        return alt_text

def main():
    setup_logging()
    data = load_data()
    gen_prompt = load_prompt(PROMPT_GEN_PATH)
    ref_prompt = load_prompt(PROMPT_REF_PATH)

    logging.info(f"[✓] Starte Verarbeitung von {len(data)} Artikeln")

    for i, item in enumerate(data, 1):
        headline = item.get("headline", "")
        abstract = item.get("abstract", "")
        caption = item.get("caption", "")
        image_url = item.get("image_url_clean", "")

        if not image_url:
            logging.warning(f"[{i}] Kein Bild gefunden – übersprungen")
            continue

        logging.info(f"[{i}] Generiere Alt-Text für: {headline[:60]}...")

        initial = generate_alt_text(image_url, headline, abstract, caption, gen_prompt)
        if not initial:
            continue

        refined = refine_alt_text(initial, headline, abstract, caption, image_url, ref_prompt)

        # Datensatz-Eintrag um neue Felder ergänzen
        item["openai_alt_text_initial"] = initial
        item["openai_alt_text_refined"] = refined

        logging.info(f"[{i}] ↳ Initial: {initial[:50]}")
        logging.info(f"[{i}] ↳ Refined: {refined[:50]}")

        if i % 25 == 0:
            save_data(data[:i])
            logging.info(f"[{i}] Zwischenspeicherung durchgeführt")

    save_data(data)
    logging.info(f"[✓] Verarbeitung abgeschlossen. Ergebnisse gespeichert unter: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()