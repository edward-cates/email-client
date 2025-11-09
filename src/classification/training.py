"""Training script for BERT email classification model"""
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
from torch.nn import CrossEntropyLoss
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.classification.dataset import create_huggingface_dataset
from src.classification.model import load_model_and_tokenizer, get_num_labels

console = Console()

MODEL_WEIGHTS_PATH = Path(__file__).parent / "model_weights.pt"


class RichMetricsCallback(TrainerCallback):
    """Custom callback to display metrics in a beautiful table format"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Only show metrics table at evaluation steps
        if "eval_loss" in logs or "eval_accuracy" in logs:
            table = Table(title="📊 Evaluation Metrics", box=box.ROUNDED, show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="green", justify="right")
            
            epoch = logs.get("epoch", state.epoch)
            if epoch is not None:
                table.add_row("Epoch", f"{epoch:.2f}")
            
            if "eval_loss" in logs:
                table.add_row("Loss", f"{logs['eval_loss']:.4f}")
            if "eval_accuracy" in logs:
                table.add_row("Accuracy", f"{logs['eval_accuracy']:.4%}")
            if "eval_runtime" in logs:
                table.add_row("Runtime", f"{logs['eval_runtime']:.2f}s")
            
            console.print(table)
        
        # Show training metrics in a compact format
        elif "loss" in logs and "eval_loss" not in logs:
            step = logs.get("step", "")
            loss = logs.get("loss", 0)
            learning_rate = logs.get("learning_rate", 0)
            
            console.print(
                f"[bold blue]Step {step}[/bold blue] | "
                f"[yellow]Loss: {loss:.4f}[/yellow] | "
                f"[cyan]LR: {learning_rate:.2e}[/cyan]"
            )


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
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    patience: int = 3,
):
    """Train the BERT model
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model: BERT model to train
        tokenizer: Tokenizer for the model
        output_dir: Directory to save training outputs
        num_epochs: Maximum number of training epochs (early stopping may stop earlier)
        batch_size: Training batch size
        learning_rate: Learning rate for training
        patience: Number of epochs to wait for accuracy improvement before stopping
    """
    # Tokenize datasets
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer)
    
    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Calculate class weights to handle imbalanced dataset
    import torch
    from collections import Counter
    train_labels = [train_dataset[i]["label"].item() for i in range(len(train_dataset))]
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    # Get number of classes from model config (should be 6)
    num_classes = model.config.num_labels
    
    # Calculate inverse frequency weights
    # For classes not in training set, use average weight
    class_weights = torch.ones(num_classes) * (total_samples / num_classes)
    for label_idx, count in label_counts.items():
        if count > 0:
            class_weights[label_idx] = total_samples / (len(label_counts) * count)
    
    # Log class weights and distribution (optional debug info)
    # console.print(f"[dim]Class weights: {dict(enumerate(class_weights.tolist()))}[/dim]")
    # console.print(f"[dim]Label distribution in training set: {dict(label_counts)}[/dim]")
    
    # Create custom Trainer with weighted loss
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights.to(self.model.device) if class_weights is not None else None
        
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
    
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
        greater_is_better=True,
        warmup_steps=10,  # Add warmup to help with small datasets
        save_total_limit=2,  # Limit checkpoints
        fp16=False,  # Ensure full precision for small datasets
    )
    
    # Metrics function
    def compute_metrics(eval_pred):
        import numpy as np
        from transformers import EvalPrediction
        
        # Handle different input formats
        if isinstance(eval_pred, tuple) and len(eval_pred) == 2:
            # Tuple format: (predictions, labels)
            predictions, labels = eval_pred
        elif isinstance(eval_pred, EvalPrediction):
            # EvalPrediction is a named tuple with predictions and label_ids
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        elif hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            # Try attribute access for other object types
            predictions = eval_pred.predictions  # type: ignore
            labels = eval_pred.label_ids  # type: ignore
        else:
            raise ValueError(f"Unexpected eval_pred type: {type(eval_pred)}")
        
        # Ensure numpy arrays
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        
        # Get predicted class indices (argmax along last axis)
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=-1)
        else:
            pred_classes = predictions
        
        # Ensure labels are integers
        labels = labels.astype(int)
        pred_classes = pred_classes.astype(int)
        
        # Calculate accuracy
        correct = (pred_classes == labels).sum()
        total = len(labels)
        accuracy = correct / total if total > 0 else 0.0
        
        return {"accuracy": float(accuracy)}
    
    # Create early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=0.0,
    )
    
    # Create trainer with rich callback, weighted loss, and early stopping
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[RichMetricsCallback(), early_stopping],
    )
    
    # Train
    trainer.train()
    
    return trainer


def main():
    """Main training function: load dataset, split, train, and save"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BERT email classification model")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs to wait for accuracy improvement before stopping (default: 3)",
    )
    args = parser.parse_args()
    
    console.print(Panel.fit("[bold cyan]🚀 BERT Email Classification Training[/bold cyan]", border_style="cyan"))
    
    # Load dataset
    with console.status("[bold green]Loading dataset...", spinner="dots"):
        dataset = create_huggingface_dataset()
    
    console.print(f"[green]✓[/green] Dataset loaded: [bold]{len(dataset)}[/bold] samples")
    
    if len(dataset) == 0:
        console.print("[bold red]✗[/bold red] Dataset is empty!")
        raise ValueError(
            "Dataset is empty. No emails with custom labels found. "
            "Please ensure you have emails with at least one custom label "
            "(marketing, noti, event, newsletter, or direct) in your Gmail accounts."
        )
    
    # Train/test split (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    split_table = Table(show_header=False, box=None, padding=(0, 2))
    split_table.add_row("[cyan]Train samples:[/cyan]", f"[bold]{len(train_dataset)}[/bold]")
    split_table.add_row("[cyan]Test samples:[/cyan]", f"[bold]{len(test_dataset)}[/bold]")
    console.print(split_table)
    
    # Load model and tokenizer
    with console.status("[bold green]Loading model and tokenizer...", spinner="dots"):
        model, tokenizer = load_model_and_tokenizer()
    
    model_table = Table(show_header=False, box=None, padding=(0, 2))
    model_table.add_row("[cyan]Model:[/cyan]", f"[bold]{model.config.name_or_path}[/bold]")
    model_table.add_row("[cyan]Labels:[/cyan]", f"[bold]{get_num_labels()}[/bold]")
    console.print(model_table)
    
    # Train
    console.print(f"\n[bold yellow]🎯 Starting training...[/bold yellow] (patience: {args.patience})\n")
    trainer = train_model(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        tokenizer=tokenizer,
        output_dir="output",
        num_epochs=50,
        batch_size=16,
        patience=args.patience,
    )
    
    # Save model weights (best model is already loaded due to load_best_model_at_end=True)
    with console.status(f"[bold green]Saving model weights to {MODEL_WEIGHTS_PATH}...", spinner="dots"):
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    
    console.print(f"\n[bold green]✓ Training complete![/bold green] Model saved to [cyan]{MODEL_WEIGHTS_PATH}[/cyan]")


if __name__ == "__main__":
    main()


