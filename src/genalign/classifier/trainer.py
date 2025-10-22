import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import List, Tuple, Dict, Any, Optional
import logging
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ClassifierTrainer:
    """Trainer for the RoBERTa classifier."""
    
    def __init__(
        self,
        classifier,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = "auto"
    ):
        """
        Initialize the classifier trainer.
        
        Args:
            classifier: RoBERTaClassifier instance
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use for training
        """
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.device = classifier.device
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
    
    def _setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Setup optimizer
        self.optimizer = AdamW(
            self.classifier.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(
        self, 
        dataloader: DataLoader,
        epoch: int,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Train the classifier for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            gradient_accumulation_steps: Number of steps to accumulate gradients
        
        Returns:
            Dictionary containing training metrics
        """
        self.classifier.enable_training_mode()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch data
            texts = [item["text"] for item in batch]
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
            
            # Tokenize texts
            inputs = self.classifier.tokenize_texts(texts)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.classifier.model(**inputs)
            logits = outputs.logits
            
            # Compute loss
            loss = self.criterion(logits, labels)
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.classifier.model.parameters(), 
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Compute metrics
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).sum().item()
            
            total_loss += loss.item() * gradient_accumulation_steps
            total_correct += correct
            total_samples += labels.size(0)
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                "acc": f"{correct / labels.size(0):.4f}"
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the classifier for one epoch.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Dictionary containing validation metrics
        """
        self.classifier.enable_eval_mode()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Prepare batch data
                texts = [item["text"] for item in batch]
                labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
                
                # Tokenize texts
                inputs = self.classifier.tokenize_texts(texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.classifier.model(**inputs)
                logits = outputs.logits
                
                # Compute loss
                loss = self.criterion(logits, labels)
                
                # Compute metrics
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute epoch metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "predictions": all_predictions,
            "labels": all_labels
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        save_best_model: bool = True,
        save_path: str = None
    ) -> Dict[str, List[float]]:
        """
        Train the classifier for multiple epochs.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Number of steps to accumulate gradients
            save_best_model: Whether to save the best model
            save_path: Path to save the best model
        
        Returns:
            Dictionary containing training history
        """
        # Setup optimizer and scheduler
        num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self._setup_optimizer_and_scheduler(num_training_steps)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        logging.info(f"Starting training for {num_epochs} epochs")
        logging.info(f"Total training steps: {num_training_steps}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(
                train_dataloader, epoch + 1, gradient_accumulation_steps
            )
            
            # Validation
            val_metrics = self.validate_epoch(val_dataloader)
            
            # Update history
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["train_accuracy"].append(train_metrics["accuracy"])
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["val_accuracy"].append(val_metrics["accuracy"])
            
            # Log metrics
            epoch_time = time.time() - start_time
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if save_best_model and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_model_state = self.classifier.model.state_dict().copy()
                
                if save_path:
                    self.classifier.save_model(save_path)
                    logging.info(f"Best model saved to {save_path}")
        
        # Restore best model
        if save_best_model and best_model_state is not None:
            self.classifier.model.load_state_dict(best_model_state)
            logging.info("Best model restored")
        
        return self.training_history
    
    def evaluate_on_golden_data(
        self, 
        golden_texts: List[str], 
        golden_labels: List[int],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate the classifier on golden data.
        
        Args:
            golden_texts: List of golden text samples
            golden_labels: List of golden labels
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.classifier.enable_eval_mode()
        
        # Get predictions
        predictions, probabilities = self.classifier.predict(
            golden_texts, batch_size, return_probabilities=True
        )
        
        # Compute metrics
        accuracy = accuracy_score(golden_labels, predictions)
        
        # Compute cross-entropy loss
        golden_labels_tensor = torch.tensor(golden_labels, dtype=torch.long)
        loss = self.criterion(probabilities, golden_labels_tensor)
        
        # Classification report
        report = classification_report(
            golden_labels, 
            predictions, 
            target_names=[self.classifier.id_to_label[i] for i in range(self.classifier.num_labels)],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(golden_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "loss": loss.item(),
            "predictions": predictions,
            "probabilities": probabilities,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get the training history."""
        return self.training_history.copy()
    
    def reset_training_history(self):
        """Reset the training history."""
        self.training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
