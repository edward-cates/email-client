"""Training script for BERT email classification model"""
from pathlib import Path
import torch
import numpy as np
from transformers import TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
from torch.nn import CrossEntropyLoss
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from sklearn.metrics import confusion_matrix

from src.classification.dataset import create_huggingface_dataset
from src.classification.model import load_model_and_tokenizer, get_num_labels, load_labels, get_ml_labels

console = Console()

MODEL_DIR = Path(__file__).parent / "model"


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
    use_mps = torch.backends.mps.is_available()
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
        use_mps_device=use_mps,  # Use Apple Silicon GPU if available
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


def _display_confusion_matrix_and_confused_samples(trainer, eval_dataset, tokenizer, original_texts):
    """Display confusion matrix and top 3 most confused samples"""
    console.print("\n[bold cyan]📊 Computing predictions and confusion matrix...[/bold cyan]")
    
    # Ensure dataset format is correct for prediction
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Get predictions
    predictions = trainer.predict(eval_dataset)
    pred_logits = predictions.predictions
    true_labels = predictions.label_ids
    
    # Get predicted classes
    pred_classes = np.argmax(pred_logits, axis=-1)
    
    # Compute softmax probabilities for confidence scores
    probs = torch.nn.functional.softmax(torch.tensor(pred_logits), dim=-1).numpy()
    pred_confidences = np.max(probs, axis=-1)
    
    # Load label names and filter to only ML labels
    labels_data = load_labels()
    all_label_names = [label["name"] for label in labels_data]
    ml_labels = get_ml_labels()
    label_names = [label["name"] for label in ml_labels]
    
    # Get indices of ML labels only
    label_indices = [i for i, name in enumerate(all_label_names) if name in label_names]
    
    # Compute confusion matrix with all labels, then extract only ML label rows/columns
    cm = confusion_matrix(true_labels, pred_classes, labels=range(len(all_label_names)))
    cm = cm[np.ix_(label_indices, label_indices)]
    
    # Display confusion matrix
    console.print("\n[bold yellow]📈 Confusion Matrix[/bold yellow]")
    cm_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    cm_table.add_column("Actual \\ Predicted", style="cyan", no_wrap=True)
    for label_name in label_names:
        cm_table.add_column(label_name, style="green", justify="right")
    
    for i, label_name in enumerate(label_names):
        row_values = []
        for j in range(len(label_names)):
            value = str(cm[i, j])
            # Color diagonal (correct predictions) green, others red
            if i == j:
                row_values.append(f"[green]{value}[/green]")
            else:
                row_values.append(f"[red]{value}[/red]")
        row = [label_name] + row_values
        cm_table.add_row(*row)
    
    console.print(cm_table)
    
    # Find incorrect predictions with highest confidence
    incorrect_mask = pred_classes != true_labels
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(incorrect_indices) == 0:
        console.print("\n[bold green]🎉 Perfect predictions! No incorrect samples found.[/bold green]")
        return
    
    # Get confidence scores for incorrect predictions
    incorrect_confidences = pred_confidences[incorrect_indices]
    
    # Sort by confidence (descending) and get top 3
    top_confused_indices = incorrect_indices[np.argsort(incorrect_confidences)[::-1]][:3]
    
    # Display top 3 confused samples
    console.print("\n[bold yellow]🔍 Top 3 Most Confused Samples[/bold yellow]")
    
    for idx, sample_idx in enumerate(top_confused_indices, 1):
        true_label = int(true_labels[sample_idx])
        pred_label = int(pred_classes[sample_idx])
        confidence = float(pred_confidences[sample_idx])
        
        # Get the original text
        text = original_texts[sample_idx]
        
        # Truncate text if too long
        if len(text) > 200:
            text = text[:200] + "..."
        
        panel_content = f"[bold]True Label:[/bold] [green]{all_label_names[true_label]}[/green]\n"
        panel_content += f"[bold]Predicted:[/bold] [red]{all_label_names[pred_label]}[/red]\n"
        panel_content += f"[bold]Confidence:[/bold] {confidence:.2%}\n\n"
        panel_content += f"[dim]{text}[/dim]"
        
        console.print(Panel(panel_content, title=f"Sample {idx}", border_style="yellow"))


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
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50)",
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
    
    # Store original text before tokenization for displaying confused samples
    test_texts = test_dataset["text"]
    
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
    console.print(f"\n[bold yellow]🎯 Starting training...[/bold yellow] (max_epochs: {args.max_epochs}, patience: {args.patience})\n")
    trainer = train_model(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model=model,
        tokenizer=tokenizer,
        output_dir="output",
        num_epochs=args.max_epochs,
        batch_size=16,
        patience=args.patience,
    )
    
    # Save model using transformers' save_pretrained (best model is already loaded due to load_best_model_at_end=True)
    with console.status(f"[bold green]Saving model to {MODEL_DIR}...", spinner="dots"):
        MODEL_DIR.mkdir(exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
    
    console.print(f"\n[bold green]✓ Training complete![/bold green] Model saved to [cyan]{MODEL_DIR}[/cyan]")
    
    # Compute and display confusion matrix and confused samples
    # Use trainer's eval_dataset which is already properly formatted
    _display_confusion_matrix_and_confused_samples(trainer, trainer.eval_dataset, tokenizer, test_texts)


if __name__ == "__main__":
    main()


