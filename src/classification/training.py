"""Training script for BERT email classification model"""
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from src.classification.dataset import create_huggingface_dataset
from src.classification.model import load_model_and_tokenizer, get_num_labels

MODEL_WEIGHTS_DIR = Path(__file__).parent / "model_weights"
MODEL_WEIGHTS_PATH = MODEL_WEIGHTS_DIR / "model_weights.pt"


def tokenize_dataset(dataset: Dataset, tokenizer):
    """Tokenize the dataset for BERT input"""
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    
    return dataset.map(tokenize, batched=True)


def train_model(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    model,
    tokenizer,
    output_dir: str | Path = "output",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """Train the BERT model
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model: BERT model to train
        tokenizer: Tokenizer for the model
        output_dir: Directory to save training outputs
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
    """
    # Tokenize datasets
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer)
    
    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # Metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    return trainer


def main():
    """Main training function: load dataset, split, train, and save"""
    # Load dataset
    print("Loading dataset...")
    dataset = create_huggingface_dataset()
    print(f"Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) == 0:
        raise ValueError(
            "Dataset is empty. No emails with custom labels found. "
            "Please ensure you have emails with at least one custom label "
            "(marketing, boring noti, event, newsletter, or direct) in your Gmail accounts."
        )
    
    # Train/test split (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded: {model.config.name_or_path}")
    print(f"Number of labels: {get_num_labels()}")
    
    # Train
    print("Starting training...")
    trainer = train_model(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        tokenizer=tokenizer,
        output_dir="output",
        num_epochs=3,
        batch_size=16,
    )
    
    # Save model weights
    print(f"Saving model weights to {MODEL_WEIGHTS_PATH}...")
    MODEL_WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print("Training complete!")


if __name__ == "__main__":
    main()

