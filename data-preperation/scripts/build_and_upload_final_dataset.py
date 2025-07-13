import os
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from pathlib import Path

# 1️⃣ Load environment variables and Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set your Hugging Face token in the .env file as HF_TOKEN."

# 2️⃣ Configurations
DATASET_NAME_RAW = "Alex23o4/n24news_sample_synthetic_alttext"
DATASET_NAME_REDUCED = "Alex23o4/n24news_sample_synthetic_alttext_reduced"
LOCAL_SAVE_PATH = "processed/n24news_reduced"

# Hugging Face Login
login(token=HF_TOKEN)

# 3️⃣ Load Full Dataset
print("📥 Loading full dataset...")
raw_ds = load_dataset(DATASET_NAME_RAW)

# 4️⃣ Fields to Retain
FIELDS_TO_KEEP = [
    "abstract",
    "caption",
    "headline",
    "image_id",
    "openai_alt_text_refined",
    "section",
    "section_label",
    "image_url_clean",
]

# 5️⃣ Transformation Function to Reduce Fields
def reduce_example(example):
    return {key: example[key] for key in FIELDS_TO_KEEP}

print("🧹 Reducing dataset...")
reduced_ds = raw_ds.map(
    reduce_example,
    remove_columns=raw_ds["train"].column_names  # Remove all other columns
)

# 6️⃣ Show Example
print("\n✅ Example after reduction:")
print(reduced_ds["train"][0])

# 7️⃣ (Optional) Save Locally
if LOCAL_SAVE_PATH:
    print(f"💾 Saving dataset locally to {LOCAL_SAVE_PATH}...")
    os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)
    reduced_ds.save_to_disk(LOCAL_SAVE_PATH)

# 8️⃣ (Optional) Push to Hugging Face Hub
push_to_hub = True  # Set to False if you don't want to push

if push_to_hub:
    print(f"🚀 Pushing reduced dataset to Hugging Face Hub as '{DATASET_NAME_REDUCED}'...")
    reduced_ds.push_to_hub(DATASET_NAME_REDUCED)

print("\n🎉 All done! Reduced dataset is ready.")
