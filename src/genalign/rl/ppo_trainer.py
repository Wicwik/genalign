import torch
import torch.nn as nn
from trl import PPOTrainer, PPOConfig
from transformers import DataCollatorForLanguageModeling
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from tqdm import tqdm
import time


class PPOTrainerWrapper:
    """Wrapper for PPO training using TRL library."""
    
    def __init__(
        self,
        generator,
        reward_model,
        config: Dict[str, Any],
        device: str = "auto",
        logger: logging.Logger = None
    ):
        """
        Initialize the PPO trainer.
        
        Args:
            generator: LlamaGenerator instance
            reward_model: RewardModel instance
            config: Configuration dictionary
            device: Device to use for training
        """
        self.generator = generator
        self.reward_model = reward_model
        self.config = config
        self.device = generator.device
        self.logger = logger
        
        # PPO configuration
        ppo_config = PPOConfig(
            model_name=generator.model_name,
            learning_rate=config.get("learning_rate", 1e-5),
            batch_size=config.get("batch_size", 4),
            mini_batch_size=config.get("mini_batch_size", 2),
            ppo_epochs=config.get("ppo_epochs", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            cliprange=config.get("cliprange", 0.2),
            cliprange_value=config.get("cliprange_value", 0.2),
            vf_coef=config.get("vf_coef", 0.1),
            ent_coef=config.get("ent_coef", 0.01),
            target_kl=config.get("target_kl", 0.01),
            log_with=config.get("log_with", "tensorboard"),
            tracker_project_name=config.get("tracker_project_name", "genalign-ppo"),
            tracker_run_name=config.get("tracker_run_name", "ppo_run"),
            save_freq=config.get("save_freq", 10),
            output_dir=config.get("output_dir", "./outputs/ppo"),
            seed=config.get("seed", 42)
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=generator.model,
            ref_model=None,  # We'll use the same model as reference initially
            tokenizer=generator.tokenizer,
            dataset=None,  # We'll provide data dynamically
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=generator.tokenizer,
                mlm=False
            )
        )
        
        # Training history
        self.training_history = {
            "rewards": [],
            "kl_divergences": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": []
        }
        
        self.logger.info("PPO trainer initialized")
    
    def compute_reward_for_batch(
        self,
        generated_texts: List[str],
        generated_labels: List[int],
        classifier,
        golden_texts: List[str],
        golden_labels: List[int]
    ) -> List[float]:
        """
        Compute rewards for a batch of generated texts.
        
        Args:
            generated_texts: List of generated text samples
            generated_labels: List of generated labels
            classifier: RoBERTaClassifier instance
            golden_texts: List of golden text samples
            golden_labels: List of golden labels
        
        Returns:
            List of reward values
        """
        # Train classifier on generated data
        from ..classifier.trainer import ClassifierTrainer
        from ..data.dataset import create_dataloader
        
        # Create synthetic dataset
        from ..data.dataset import SyntheticDataset
        synthetic_dataset = SyntheticDataset()
        synthetic_dataset.add_samples(generated_texts, generated_labels, 
                                    [classifier.id_to_label[label] for label in generated_labels], 0)
        
        # Create dataloader
        train_dataloader = create_dataloader(synthetic_dataset, batch_size=8, shuffle=True)
        
        # Create trainer and train
        trainer = ClassifierTrainer(classifier)
        trainer.train(train_dataloader, train_dataloader, num_epochs=2, save_best_model=False)
        
        # Evaluate on golden data
        eval_results = trainer.evaluate_on_golden_data(golden_texts, golden_labels)
        golden_loss = eval_results["loss"]
        
        # Compute distance metrics
        from ..metrics.distances import DistanceCalculator
        distance_calculator = DistanceCalculator()
        distance_metrics = distance_calculator.compute_distance_metrics(
            generated_texts, generated_labels, classifier
        )
        
        # Compute reward
        reward_info = self.reward_model.compute_reward(
            inter_class_distance=distance_metrics["avg_inter_class_distance"],
            intra_class_distance=distance_metrics["avg_intra_class_distance"],
            golden_loss=golden_loss
        )
        
        # Return the same reward for all samples in the batch
        return [reward_info["reward"]] * len(generated_texts)
    
    def prepare_ppo_data(
        self,
        prompts: List[str],
        generated_texts: List[str],
        rewards: List[float]
    ) -> Dict[str, Any]:
        """
        Prepare data for PPO training.
        
        Args:
            prompts: List of input prompts
            generated_texts: List of generated texts
            rewards: List of reward values
        
        Returns:
            Dictionary containing PPO training data
        """
        # Tokenize prompts and generated texts
        prompt_tokens = self.generator.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        generated_tokens = self.generator.tokenizer(
            generated_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Create PPO data format
        ppo_data = {
            "input_ids": generated_tokens["input_ids"],
            "attention_mask": generated_tokens["attention_mask"],
            "query": prompt_tokens["input_ids"],
            "query_attention_mask": prompt_tokens["attention_mask"],
            "rewards": torch.tensor(rewards, dtype=torch.float32)
        }
        
        return ppo_data
    
    def train_step(
        self,
        prompts: List[str],
        generated_texts: List[str],
        rewards: List[float],
        classifier,
        golden_texts: List[str],
        golden_labels: List[int]
    ) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Args:
            prompts: List of input prompts
            generated_texts: List of generated texts
            rewards: List of reward values
            classifier: RoBERTaClassifier instance
            golden_texts: List of golden text samples
            golden_labels: List of golden labels
        
        Returns:
            Dictionary containing training metrics
        """
        # Compute rewards if not provided
        if rewards is None:
            # Extract labels from generated texts (this is a simplified approach)
            # In practice, you might need to parse the generated texts to extract labels
            generated_labels = [0] * len(generated_texts)  # Placeholder
            rewards = self.compute_reward_for_batch(
                generated_texts, generated_labels, classifier, golden_texts, golden_labels
            )
        
        # Prepare PPO data
        ppo_data = self.prepare_ppo_data(prompts, generated_texts, rewards)
        
        # Convert to the format expected by TRL
        batch = {
            "input_ids": ppo_data["input_ids"],
            "attention_mask": ppo_data["attention_mask"],
            "query": ppo_data["query"],
            "query_attention_mask": ppo_data["query_attention_mask"],
            "rewards": ppo_data["rewards"]
        }
        
        # Perform PPO step
        stats = self.ppo_trainer.step([batch])
        
        # Update training history
        if stats:
            self.training_history["rewards"].append(np.mean(rewards))
            if "ppo/mean_scores" in stats:
                self.training_history["rewards"].append(stats["ppo/mean_scores"])
            if "ppo/policy/policy_loss" in stats:
                self.training_history["policy_losses"].append(stats["ppo/policy/policy_loss"])
            if "ppo/val/val_loss" in stats:
                self.training_history["value_losses"].append(stats["ppo/val/val_loss"])
            if "ppo/policy/entropy" in stats:
                self.training_history["entropies"].append(stats["ppo/policy/entropy"])
            if "ppo/policy/kl" in stats:
                self.training_history["kl_divergences"].append(stats["ppo/policy/kl"])
        
        return stats or {}
    
    def train_iteration(
        self,
        icl_examples: List[Tuple[str, int, str]],
        num_samples: int,
        classifier,
        golden_texts: List[str],
        golden_labels: List[int],
        target_classes: List[int] = None
    ) -> Dict[str, Any]:
        """
        Perform one complete training iteration.
        
        Args:
            icl_examples: List of ICL examples
            num_samples: Number of samples to generate
            classifier: RoBERTaClassifier instance
            golden_texts: List of golden text samples
            golden_labels: List of golden labels
            target_classes: List of target classes for generation
        
        Returns:
            Dictionary containing iteration results
        """
        # Generate samples
        generated_samples = self.generator.generate_samples(
            icl_examples=icl_examples,
            num_samples=num_samples,
            target_classes=target_classes
        )
        
        if not generated_samples:
            self.logger.warning("No samples generated successfully")
            return {"success": False, "error": "No samples generated"}
        
        # Extract texts and labels
        generated_texts = [sample[0] for sample in generated_samples]
        generated_labels = [sample[1] for sample in generated_samples]
        
        # Create prompts (simplified - in practice you'd use the actual prompts)
        prompts = [f"Generate a text sample for class {label}" for label in generated_labels]
        
        # Perform PPO training step
        training_stats = self.train_step(
            prompts=prompts,
            generated_texts=generated_texts,
            rewards=None,  # Will be computed internally
            classifier=classifier,
            golden_texts=golden_texts,
            golden_labels=golden_labels
        )
        
        return {
            "success": True,
            "generated_samples": generated_samples,
            "num_samples": len(generated_samples),
            "training_stats": training_stats
        }
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        self.ppo_trainer.save_model(save_path)
        self.logger.info(f"PPO model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a trained model."""
        # This would need to be implemented based on TRL's save/load functionality
        self.logger.info(f"Loading PPO model from {load_path}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get the training history."""
        return self.training_history.copy()
    
    def reset_training_history(self):
        """Reset the training history."""
        self.training_history = {
            "rewards": [],
            "kl_divergences": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": []
        }
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model configuration parameters."""
        return {
            "config": self.config,
            "device": str(self.device),
            "training_history_size": {
                key: len(values) for key, values in self.training_history.items()
            }
        }


class SimplifiedPPOTrainer:
    """Simplified PPO trainer for cases where TRL integration is complex."""
    
    def __init__(
        self,
        generator,
        reward_model,
        config: Dict[str, Any]
    ):
        """
        Initialize the simplified PPO trainer.
        
        Args:
            generator: LlamaGenerator instance
            reward_model: RewardModel instance
            config: Configuration dictionary
        """
        self.generator = generator
        self.reward_model = reward_model
        self.config = config
        
        # Training parameters
        self.learning_rate = config.get("learning_rate", 1e-5)
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.generator.model.parameters(),
            lr=self.learning_rate
        )
        
        # Training history
        self.training_history = {
            "rewards": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": []
        }
        
        self.logger.info("Simplified PPO trainer initialized")
    
    def compute_advantages(self, rewards: List[float], values: List[float]) -> List[float]:
        """Compute advantages using simple baseline."""
        advantages = []
        baseline = np.mean(rewards)
        
        for reward, value in zip(rewards, values):
            advantage = reward - baseline
            advantages.append(advantage)
        
        return advantages
    
    def train_step_simplified(
        self,
        prompts: List[str],
        generated_texts: List[str],
        rewards: List[float]
    ) -> Dict[str, float]:
        """
        Perform a simplified PPO training step.
        
        Args:
            prompts: List of input prompts
            generated_texts: List of generated texts
            rewards: List of reward values
        
        Returns:
            Dictionary containing training metrics
        """
        # This is a simplified implementation
        # In practice, you would implement proper PPO with policy and value networks
        
        # For now, we'll just do a simple policy gradient update
        self.generator.model.train()
        
        # Tokenize inputs
        inputs = self.generator.tokenizer(
            generated_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.generator.device)
        
        # Forward pass
        outputs = self.generator.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Weight the loss by rewards
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.generator.device)
        weighted_loss = loss * reward_tensor.mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update history
        self.training_history["rewards"].append(np.mean(rewards))
        self.training_history["policy_losses"].append(weighted_loss.item())
        
        return {
            "policy_loss": weighted_loss.item(),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards)
        }

