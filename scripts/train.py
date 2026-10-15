"""Fine-tune Vietnamese toxic-comment binary classifier using Hugging Face data.

Baseline dataset: tarudesu/ViCTSD
Columns: Comment (text), Toxicity (0=non-toxic, 1=toxic)
Splits: train, validation, test
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def integer_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def limit_split(split, maximum: int, seed: int):
    if maximum < 0:
        raise ValueError("Sample limits must be >= 0; 0 means all samples")
    if maximum == 0 or maximum >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(maximum))


def main() -> None:
    seed = integer_env("SEED", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_id = os.environ.get("HF_DATASET_ID", "tarudesu/ViCTSD")
    dataset_revision = os.environ.get("HF_DATASET_REVISION", "main")
    model_name = os.environ.get(
        "MODEL_NAME", "distilbert/distilbert-base-multilingual-cased"
    )
    hf_token = os.environ.get("HF_TOKEN") or None
    output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs"))
    max_length = integer_env("MAX_LENGTH", 128)
    batch_size = integer_env("BATCH_SIZE", 4)
    epochs = float(os.environ.get("EPOCHS", "2"))

    # Never put datasets, authentication tokens, or model weights in Git.
    dataset = load_dataset(
        dataset_id, revision=dataset_revision, token=hf_token
    )
    validation_name = "validation" if "validation" in dataset else "valid"
    required_splits = ("train", validation_name, "test")
    for name in required_splits:
        if name not in dataset:
            raise ValueError(f"Missing split: {name}; found {list(dataset.keys())}")
        required_columns = {"Comment", "Toxicity"}
        if not required_columns.issubset(dataset[name].column_names):
            raise ValueError(
                f"Split {name} must contain {required_columns}; "
                f"found {dataset[name].column_names}. "
                "Adapt the preprocessing before switching datasets."
            )

    raw = {
        "train": limit_split(dataset["train"], integer_env("MAX_TRAIN_SAMPLES", 0), seed),
        "validation": limit_split(
            dataset[validation_name], integer_env("MAX_EVAL_SAMPLES", 0), seed
        ),
        "test": limit_split(dataset["test"], integer_env("MAX_TEST_SAMPLES", 0), seed),
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    def preprocess(batch):
        encoded = tokenizer(
            batch["Comment"], max_length=max_length, truncation=True
        )
        encoded["labels"] = [int(label) for label in batch["Toxicity"]]
        return encoded

    prepared = {
        name: split.map(
            preprocess,
            batched=True,
            remove_columns=split.column_names,
            desc=f"Tokenizing {name}",
        )
        for name, split in raw.items()
    }

    id2label = {0: "NON_TOXIC", 1: "TOXIC"}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        token=hf_token,
    )

    def compute_metrics(prediction):
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted = np.argmax(logits, axis=-1)
        labels = prediction.label_ids
        return {
            "accuracy": accuracy_score(labels, predicted),
            "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
        }

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=prepared["train"],
        eval_dataset=prepared["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    validation_metrics = trainer.evaluate()
    # The held-out test split is evaluated only after training.
    test_output = trainer.predict(prepared["test"])
    logits = test_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted = np.argmax(logits, axis=-1)
    report = classification_report(
        test_output.label_ids,
        predicted,
        labels=[0, 1],
        target_names=["NON_TOXIC", "TOXIC"],
        output_dict=True,
        zero_division=0,
    )

    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset_id": dataset_id,
        "dataset_revision_requested": dataset_revision,
        "model_name": model_name,
        "seed": seed,
        "train_examples": len(prepared["train"]),
        "validation_examples": len(prepared["validation"]),
        "test_examples": len(prepared["test"]),
        "validation": validation_metrics,
        "test": report,
        "warning": "A subset run is a pipeline smoke test, not a research benchmark.",
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Optional: publishing needs a separate HF write token and explicit HF_MODEL_REPO_ID.
    model_repo = os.environ.get("HF_MODEL_REPO_ID", "").strip()
    if model_repo:
        if not hf_token:
            raise ValueError("HF_MODEL_REPO_ID set but HF_TOKEN is missing")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True, private=True)
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=model_repo,
            repo_type="model",
            commit_message="Upload ViCTSD baseline fine-tuning output",
        )
        print(f"Uploaded model to Hugging Face: {model_repo}")


if __name__ == "__main__":
    main()
