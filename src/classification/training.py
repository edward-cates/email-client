"""Training script for BERT email classification model"""
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer, TrainerCallback
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
    
    # Create trainer with rich callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[RichMetricsCallback()],
    )
    
    # Train
    trainer.train()
    
    return trainer


def main():
    """Main training function: load dataset, split, train, and save"""
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
            "(marketing, boring noti, event, newsletter, or direct) in your Gmail accounts."
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
    console.print("\n[bold yellow]🎯 Starting training...[/bold yellow]\n")
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
    with console.status(f"[bold green]Saving model weights to {MODEL_WEIGHTS_PATH}...", spinner="dots"):
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    
    console.print(f"\n[bold green]✓ Training complete![/bold green] Model saved to [cyan]{MODEL_WEIGHTS_PATH}[/cyan]")


if __name__ == "__main__":
    main()


