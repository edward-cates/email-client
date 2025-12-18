"""Training script for BERT email prioritization model (regression)"""
from pathlib import Path
import torch
import numpy as np
from transformers import TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.prioritization.dataset import create_huggingface_dataset
from src.prioritization.model import load_model_and_tokenizer, score_to_label, PRIORITY_LABELS

console = Console()

MODEL_DIR = Path(__file__).parent / "model"


class RichMetricsCallback(TrainerCallback):
    """Custom callback to display metrics in a beautiful table format"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        if "eval_loss" in logs or "eval_mse" in logs:
            table = Table(title="📊 Evaluation Metrics", box=box.ROUNDED, show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="green", justify="right")
            
            epoch = logs.get("epoch", state.epoch)
            if epoch is not None:
                table.add_row("Epoch", f"{epoch:.2f}")
            
            if "eval_loss" in logs:
                table.add_row("Loss (MSE)", f"{logs['eval_loss']:.4f}")
            if "eval_mae" in logs:
                table.add_row("MAE", f"{logs['eval_mae']:.4f}")
            if "eval_accuracy" in logs:
                table.add_row("Accuracy (rounded)", f"{logs['eval_accuracy']:.2%}")
            if "eval_runtime" in logs:
                table.add_row("Runtime", f"{logs['eval_runtime']:.2f}s")
            
            console.print(table)
        
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
    output_dir: str | Path = "output_prioritization",
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    patience: int = 3,
):
    """Train the BERT regression model for prioritization
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model: BERT model to train
        tokenizer: Tokenizer for the model
        output_dir: Directory to save training outputs
        num_epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        patience: Number of epochs to wait for improvement before stopping
    """
    from collections import Counter
    
    # Tokenize datasets
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer)
    
    # Set format for PyTorch - label is float for regression
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Calculate inverse frequency weights for each priority level
    # This addresses class imbalance where rare priorities (like p1) get higher weights
    train_labels = [float(train_dataset[i]["label"]) for i in range(len(train_dataset))]
    # Round to get discrete priority levels (4=p1, 3=p2, 2=p3, 1=p4)
    train_priorities = [round(label) for label in train_labels]
    priority_counts = Counter(train_priorities)
    total_samples = len(train_labels)
    
    # Weight = total / (num_classes * count_for_this_class)
    # Higher weight for rare classes
    num_priority_levels = 4
    priority_weights = {}
    for priority in [1, 2, 3, 4]:
        count = priority_counts.get(priority, 0)
        if count > 0:
            priority_weights[priority] = total_samples / (num_priority_levels * count)
        else:
            priority_weights[priority] = 1.0
    
    # Log the weights
    console.print("\n[dim]Sample weights (inverse frequency):[/dim]")
    for priority, label in [(4, "p1"), (3, "p2"), (2, "p3"), (1, "p4")]:
        count = priority_counts.get(priority, 0)
        weight = priority_weights[priority]
        console.print(f"[dim]  {label}: count={count}, weight={weight:.2f}[/dim]")
    console.print()
    
    # Create sample weights tensor (one weight per training sample)
    sample_weights = torch.tensor([priority_weights[round(label)] for label in train_labels], dtype=torch.float32)
    
    # Custom Trainer with weighted MSE loss
    class WeightedMSETrainer(Trainer):
        def __init__(self, sample_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._sample_weights = sample_weights
            self._current_indices = None
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            
            # Standard MSE (HF's default for regression)
            # We apply weighting based on the label values (priority levels)
            # Since we can't track exact indices in batches, weight by label value
            weights = torch.tensor(
                [priority_weights[round(label.item())] for label in labels],
                device=logits.device,
                dtype=logits.dtype
            )
            
            # Weighted MSE: mean of (weight * (pred - label)^2)
            squared_errors = (logits - labels) ** 2
            weighted_loss = (weights * squared_errors).mean()
            
            return (weighted_loss, outputs) if return_outputs else weighted_loss
    
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
        metric_for_best_model="loss",  # Lower loss is better for regression
        greater_is_better=False,
        warmup_steps=10,
        save_total_limit=2,
        fp16=False,
    )
    
    def compute_metrics(eval_pred):
        """Compute regression metrics"""
        from transformers import EvalPrediction
        
        if isinstance(eval_pred, tuple) and len(eval_pred) == 2:
            predictions, labels = eval_pred
        elif isinstance(eval_pred, EvalPrediction):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        
        predictions = np.asarray(predictions).flatten()
        labels = np.asarray(labels).flatten()
        
        # MSE
        mse = np.mean((predictions - labels) ** 2)
        
        # MAE
        mae = np.mean(np.abs(predictions - labels))
        
        # Accuracy when rounded to nearest priority level
        pred_rounded = np.clip(np.round(predictions), 1, 4)
        labels_rounded = np.round(labels)
        accuracy = np.mean(pred_rounded == labels_rounded)
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "accuracy": float(accuracy),
        }
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=0.0,
    )
    
    trainer = WeightedMSETrainer(
        sample_weights=sample_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[RichMetricsCallback(), early_stopping],
    )
    
    trainer.train()
    
    return trainer


def _display_predictions_summary(trainer, eval_dataset, original_texts, original_labels):
    """Display prediction summary and examples of errors"""
    console.print("\n[bold cyan]📊 Computing predictions...[/bold cyan]")
    
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    predictions = trainer.predict(eval_dataset)
    pred_scores = predictions.predictions.flatten()
    true_scores = predictions.label_ids.flatten()
    
    # Calculate per-class accuracy
    console.print("\n[bold yellow]📈 Per-Priority Accuracy[/bold yellow]")
    
    accuracy_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    accuracy_table.add_column("Priority", style="cyan")
    accuracy_table.add_column("Count", style="green", justify="right")
    accuracy_table.add_column("Correct", style="green", justify="right")
    accuracy_table.add_column("Accuracy", style="green", justify="right")
    
    for priority, score in [("p1", 4.0), ("p2", 3.0), ("p3", 2.0), ("p4", 1.0)]:
        mask = np.isclose(true_scores, score)
        count = np.sum(mask)
        if count > 0:
            pred_rounded = np.clip(np.round(pred_scores[mask]), 1, 4)
            correct = np.sum(pred_rounded == score)
            acc = correct / count
            accuracy_table.add_row(priority, str(count), str(correct), f"{acc:.2%}")
        else:
            accuracy_table.add_row(priority, "0", "-", "-")
    
    console.print(accuracy_table)
    
    # Find worst predictions (largest error)
    errors = np.abs(pred_scores - true_scores)
    worst_indices = np.argsort(errors)[::-1][:3]
    
    if errors[worst_indices[0]] > 0.5:
        console.print("\n[bold yellow]🔍 Top 3 Worst Predictions[/bold yellow]")
        
        for idx, sample_idx in enumerate(worst_indices, 1):
            true_score = float(true_scores[sample_idx])
            pred_score = float(pred_scores[sample_idx])
            true_label = score_to_label(true_score)
            pred_label = score_to_label(pred_score)
            
            text = original_texts[sample_idx]
            if len(text) > 200:
                text = text[:200] + "..."
            
            panel_content = f"[bold]True:[/bold] [green]{true_label}[/green] (score: {true_score:.1f})\n"
            panel_content += f"[bold]Predicted:[/bold] [red]{pred_label}[/red] (score: {pred_score:.2f})\n"
            panel_content += f"[bold]Error:[/bold] {errors[sample_idx]:.2f}\n\n"
            panel_content += f"[dim]{text}[/dim]"
            
            console.print(Panel(panel_content, title=f"Sample {idx}", border_style="yellow"))


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BERT email prioritization model")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs to wait for improvement before stopping (default: 3)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50)",
    )
    args = parser.parse_args()
    
    console.print(Panel.fit("[bold cyan]🚀 BERT Email Prioritization Training[/bold cyan]", border_style="cyan"))
    
    with console.status("[bold green]Loading dataset...", spinner="dots"):
        dataset = create_huggingface_dataset()
    
    console.print(f"[green]✓[/green] Dataset loaded: [bold]{len(dataset)}[/bold] samples")
    
    if len(dataset) == 0:
        console.print("[bold red]✗[/bold red] Dataset is empty!")
        raise ValueError(
            "Dataset is empty. No emails with priority labels (p1, p2, p3, p4) found. "
            "Please label some emails with priority labels first."
        )
    
    # Train/test split (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    test_texts = test_dataset["text"]
    test_labels = test_dataset["label"]
    
    split_table = Table(show_header=False, box=None, padding=(0, 2))
    split_table.add_row("[cyan]Train samples:[/cyan]", f"[bold]{len(train_dataset)}[/bold]")
    split_table.add_row("[cyan]Test samples:[/cyan]", f"[bold]{len(test_dataset)}[/bold]")
    console.print(split_table)
    
    with console.status("[bold green]Loading model and tokenizer...", spinner="dots"):
        model, tokenizer = load_model_and_tokenizer()
    
    model_table = Table(show_header=False, box=None, padding=(0, 2))
    model_table.add_row("[cyan]Model:[/cyan]", f"[bold]{model.config.name_or_path}[/bold]")
    model_table.add_row("[cyan]Task:[/cyan]", "[bold]Regression (num_labels=1)[/bold]")
    console.print(model_table)
    
    console.print(f"\n[bold yellow]🎯 Starting training...[/bold yellow] (max_epochs: {args.max_epochs}, patience: {args.patience})\n")
    trainer = train_model(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        tokenizer=tokenizer,
        output_dir="output_prioritization",
        num_epochs=args.max_epochs,
        batch_size=16,
        patience=args.patience,
    )
    
    with console.status(f"[bold green]Saving model to {MODEL_DIR}...", spinner="dots"):
        MODEL_DIR.mkdir(exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
    
    console.print(f"\n[bold green]✓ Training complete![/bold green] Model saved to [cyan]{MODEL_DIR}[/cyan]")
    
    _display_predictions_summary(trainer, trainer.eval_dataset, test_texts, test_labels)


if __name__ == "__main__":
    main()

