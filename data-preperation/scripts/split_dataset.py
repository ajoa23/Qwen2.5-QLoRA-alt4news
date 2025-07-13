from dotenv import load_dotenv
import os
from datasets import load_dataset, DatasetDict, ClassLabel, Features
import wandb
from huggingface_hub import login

# Load .env variables
load_dotenv()

# Configurations
DATASET_JSON = "src/data/raw/full_sampled_with_alttext_augmented.json"
PROCESSED_DATA_DIR = "src/data/processed"
SEED = int(os.getenv("SEED", 42))
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

#W&B Configurations
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "qwen-vlm-alttext")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_RUN_NAME = "n24_split"

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1, "Train, validation, and test ratios must sum to 1."

def main():
    # Load the dataset
    dataset = load_dataset(
        "json",
        data_files=DATASET_JSON,
        split="train",
    )

    print(f"Loaded {dataset.num_rows} samples.")

    # maps secions to integers
    unique_sections = sorted(dataset.unique("section"))
    section_label = ClassLabel(names=unique_sections)

    # Convert the 'section' column to numeric labels
    dataset = dataset.map(
        lambda example: {"section_label": section_label.str2int(example["section"])},
    )

    # Add the section_label column to the dataset
    dataset = dataset.cast(
        Features({
            **dataset.features,
            "section_label": section_label
        })
    )
    
    # First Split
    split_results = dataset.train_test_split(
       test_size=TEST_RATIO + VAL_RATIO,
       stratify_by_column="section_label",
       seed=SEED,
    )

    train_dataset = split_results["train"]
    test_val_dataset = split_results["test"]

    # Second Split
    # Split the test_val_dataset into validation and test sets
    val_test_split = test_val_dataset.train_test_split(
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        stratify_by_column="section_label",
        seed=SEED,
    )
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]   

    # Check and report the number of samples in each split
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })

def save_splits(dataset_splits: DatasetDict):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for split_name, split_dataset in dataset_splits.items():
        path = os.path.join(PROCESSED_DATA_DIR, f"n24_{split_name}.arrow")
        split_dataset.save_to_disk(path)
        print(f"{split_name.capitalize()}-Split gespeichert unter: {path}")

def log_splits_to_wandb():
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="data-split")

    artifact = wandb.Artifact("n24_dataset_splits", type="dataset")
    artifact.add_dir(PROCESSED_DATA_DIR)
    wandb.log_artifact(artifact)

    wandb.finish()
    print("W&B Artifact für die Splits erstellt.")

def push_splits_to_hf(dataset_splits: DatasetDict):
    # Load your Hugging Face token from .env or environment variable
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN not found. Please set it in your .env or environment variables.")

    login(token=hf_token)

    # Push the DatasetDict directly to Hugging Face Hub
    dataset_splits.push_to_hub(
        "Alex23o4/n24news_sample_synthetic_alttext",
        private=False,  # Set to True if you want it to be private
        commit_message="Uploading processed and split dataset for training."
    )
    print("Dataset successfully pushed to Hugging Face Hub.")


if __name__ == "__main__":
    splits = main()
    save_splits(splits)
    log_splits_to_wandb()
    push_splits_to_hf(splits)