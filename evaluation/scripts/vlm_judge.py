import json
import os
import logging
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset

# --------------------------------------------------------------------
# Setup & Konfiguration
# --------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_PATH = "data/processed/merged_predictions_with_no_context.json"
OUTPUT_PATH = "data/processed/full_sampled_with_judging.json"
LOG_PATH = f'logs/judging_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

client = OpenAI(api_key=OPENAI_API_KEY)

# Modelle und ihre Feldnamen
variants = {
    "generated_baseline": "Baseline mit Kontext",
    "generated_baseline_no_context": "Baseline ohne Kontext",
    "generated_finetuned": "Fine-Tuned mit Kontext",
    "generated_finetuned_no_context": "Fine-Tuned ohne Kontext"
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler()
        ]
    )

def load_data(path):
    dataset = load_dataset("json", data_files=path, split="train")
    # Convert Dataset to list of dictionaries
    return [dict(item) for item in dataset]

def save_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def judge_alt_text(entry, variant_key, variant_label):
    alt_text = entry.get(variant_key)
    if not alt_text:
        return None

    # 1. System-Prompt  ─ mit harten Regeln + JSON-Schema
    system_content = (
        "You are an accessibility auditor for news-image alt texts.\n"
        "Rate each criterion on a 1–5 Likert scale (1 = poor, 5 = excellent).\n"
        "Anchors: 1 = fails, 3 = partly meets, 5 = fully meets.\n\n"
        "Criteria:\n"
        "1. visibility_principle – describe only what is directly visible OR "
        "explicitly named in caption/headline. Penalise:\n"
        "   • speculative emotions/intentions (e.g. 'angry', 'celebrates')\n"
        "   • unseen events (future, past, off-screen)\n"
        "   • context facts that are not visually verifiable (e.g. exact location if no sign)\n"
        "2. context_relevance – context must clarify or disambiguate a visible element; "
        "irrelevant context = 1.\n"
        "3. entity_naming – reward correct, context-supported names; "
        "wrong or omitted key entity = 1.\n"
        "4. informativeness – concise, image-specific; generic = 1.\n"
        "5. redundancy_avoidance – no >30 % verbatim copy of caption/headline\n"
        "6. style_readability – clear grammar; awkward/unreadable = 1.\n\n"
        "Return ONLY valid JSON (nothing else):\n"
        "{"
        "\"visibility_principle\":<1-5>,"
        "\"context_relevance\":<1-5>,"
        "\"entity_naming\":<1-5>,"
        "\"informativeness\":<1-5>,"
        "\"redundancy_avoidance\":<1-5>,"
        "\"style_readability\":<1-5>,"
        "\"total\":<1-5>,"
        "\"justification\":\"<max 2 sentences>\""
        "}"
    )

    # 2. User-Nachricht (Bild + Kontext + Alt-Text)
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": entry["image_url_clean"], "detail": "low"}
        },
        {
            "type": "text",
            "text": (
                f"Headline: {entry['headline']}\n"
                f"Abstract: {entry['abstract']}\n"
                f"Caption: {entry['caption']}\n\n"
                f"Alt-Text candidate:\n{alt_text}\n\n"
                "IMPORTANT: Deduct points for any interpretation, emotion, symbolism or "
                "context fact that cannot be visually confirmed."
            )
        }
    ]


    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Evaluation failed for {entry.get('image_id')} variant {variant_label}: {e}")
        return None

def main():
    setup_logging()
    try:
        data = load_data(INPUT_PATH)
        logging.info(f"Loaded {len(data)} entries for judging.")

        results = []
        for i, entry in enumerate(data, 1):
            if not isinstance(entry, dict):
                logging.warning(f"Skipping entry {i}: not a dictionary")
                continue
                
            image_id = entry.get("image_id")
            logging.info(f"[{i}/{len(data)}] Judging entry {image_id}")
            
            # Create a new dictionary for results
            result_entry = entry.copy()
            for key, label in variants.items():
                result = judge_alt_text(entry, key, label)
                result_entry[f"judging_{key}"] = result
                logging.info(f" → {label}: {result['entity_naming'] if result else 'Failed'}")
            
            results.append(result_entry)
            
            # Periodisches Speichern
            if i % 25 == 0:
                save_data(results, OUTPUT_PATH)
                logging.info(f"[{i}] Progress saved.")
        
        save_data(results, OUTPUT_PATH)
        logging.info(f"Judging completed. Results saved to {OUTPUT_PATH}")
    
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
