import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from typing import List, Tuple, Dict, Any, Optional
import logging
import os


class Classifier:
    """Classifier for text classification."""
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 2,
        cache_dir: str = "./cache",
        device: str = "auto",
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the model
            num_labels: Number of classification labels
            cache_dir: Directory to cache the model
            device: Device to use ("auto", "cuda", "cpu")
            dtype: Torch data type
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        self.device = self._get_device(device)
        self.dtype = dtype
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load the model and tokenizer."""
        self.logger.info(f"Loading classifier: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            cache_dir=self.cache_dir,
            dtype=self.dtype
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        self.logger.info(f"Classifier loaded successfully on {self.device}")
    
    def set_label_mapping(self, label_to_id: Dict[str, int], id_to_label: Dict[int, str]):
        """Set the label mapping for the classifier."""
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label
        self.num_labels = len(label_to_id)
    
    def tokenize_texts(
        self, 
        texts: List[str], 
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
        
        Returns:
            Dictionary containing tokenized inputs
        """
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
    
    def predict(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        return_probabilities: bool = False
    ) -> Tuple[List[int], Optional[torch.Tensor]]:
        """
        Predict labels for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for prediction
            return_probabilities: Whether to return prediction probabilities
        
        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        self.enable_eval_mode()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenize_texts(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().tolist())
                if return_probabilities:
                    all_probabilities.append(probabilities.cpu())
        
        if return_probabilities and all_probabilities:
            all_probabilities = torch.cat(all_probabilities, dim=0)
            return all_predictions, all_probabilities
        else:
            return all_predictions, None
    
    def evaluate(
        self, 
        texts: List[str], 
        true_labels: List[int],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate the classifier on a dataset.
        
        Args:
            texts: List of text strings
            true_labels: List of true label IDs
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions, probabilities = self.predict(texts, batch_size, return_probabilities=True)
        
        # Compute accuracy
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(true_labels)
        
        # Compute cross-entropy loss
        true_labels_tensor = torch.tensor(true_labels, dtype=torch.long)
        loss = nn.CrossEntropyLoss()(probabilities, true_labels_tensor)
        
        return {
            "accuracy": accuracy,
            "loss": loss.item(),
            "predictions": predictions,
            "probabilities": probabilities
        }
    
    def get_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        layer_index: int = -1
    ) -> torch.Tensor:
        """
        Extract embeddings from the model.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            layer_index: Which layer to extract embeddings from (-1 for last layer)
        
        Returns:
            Tensor containing embeddings
        """
        self.enable_eval_mode()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenize_texts(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model.roberta(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                # Extract embeddings from specified layer
                embeddings = hidden_states[layer_index]
                
                # Pool embeddings (mean pooling over sequence length)
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_embeddings = sum_embeddings / sum_mask
                
                all_embeddings.append(pooled_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def save_model(self, save_path: str):
        """Save the model and tokenizer."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save label mappings
        import json
        with open(os.path.join(save_path, "label_mappings.json"), "w") as f:
            json.dump({
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label,
                "num_labels": self.num_labels
            }, f, indent=2)
        
        self.logger.info(f"Classifier saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a saved model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            dtype=self.dtype
        )
        self.model = self.model.to(self.device)
        
        # Load label mappings
        import json
        mapping_path = os.path.join(load_path, "label_mappings.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mappings = json.load(f)
                self.label_to_id = mappings["label_to_id"]
                self.id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}
                self.num_labels = mappings["num_labels"]
        
        self.logger.info(f"Classifier loaded from {load_path}")
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model configuration parameters."""
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label
        }
    
    def enable_training_mode(self):
        """Enable training mode for the model."""
        self.model.train()
    
    def enable_eval_mode(self):
        """Enable evaluation mode for the model."""
        self.model.eval()
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

