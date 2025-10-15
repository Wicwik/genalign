import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging


class RewardModel:
    """Reward model that combines golden loss and distributional quality metrics."""
    
    def __init__(
        self,
        inter_class_weight: float = 1.0,
        intra_class_weight: float = 1.0,
        golden_loss_weight: float = 1.0,
        use_min_max_normalization: bool = True,
        clip_rewards: bool = True,
        clip_value: float = 5.0,
        reward_history_size: int = 100
    ):
        """
        Initialize the reward model.
        
        Args:
            inter_class_weight: Weight for inter-class distance component
            intra_class_weight: Weight for intra-class distance component
            golden_loss_weight: Weight for golden loss component
            use_min_max_normalization: Whether to use min-max normalization
            clip_rewards: Whether to clip reward values
            clip_value: Maximum absolute value for clipped rewards
            reward_history_size: Size of history for normalization
        """
        self.inter_class_weight = inter_class_weight
        self.intra_class_weight = intra_class_weight
        self.golden_loss_weight = golden_loss_weight
        self.use_min_max_normalization = use_min_max_normalization
        self.clip_rewards = clip_rewards
        self.clip_value = clip_value
        self.reward_history_size = reward_history_size
        
        # History for normalization
        self.inter_class_history = []
        self.intra_class_history = []
        self.golden_loss_history = []
        self.reward_history = []
        
        # Statistics for normalization
        self.inter_class_stats = {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1}
        self.intra_class_stats = {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1}
        self.golden_loss_stats = {"min": 0.0, "max": 2.0, "mean": 1.0, "std": 0.5}
    
    def update_weights(
        self, 
        inter_class_weight: float = None,
        intra_class_weight: float = None,
        golden_loss_weight: float = None
    ):
        """Update the component weights."""
        if inter_class_weight is not None:
            self.inter_class_weight = inter_class_weight
        if intra_class_weight is not None:
            self.intra_class_weight = intra_class_weight
        if golden_loss_weight is not None:
            self.golden_loss_weight = golden_loss_weight
        
        logging.info(f"Updated weights - Inter: {self.inter_class_weight}, "
                    f"Intra: {self.intra_class_weight}, Golden: {self.golden_loss_weight}")
    
    def normalize_component(
        self, 
        value: float, 
        component_name: str, 
        stats: Dict[str, float]
    ) -> float:
        """
        Normalize a component value using min-max normalization.
        
        Args:
            value: Value to normalize
            component_name: Name of the component for logging
            stats: Statistics for normalization
        
        Returns:
            Normalized value
        """
        if not self.use_min_max_normalization:
            return value
        
        # Use z-score normalization if min-max range is too small
        if stats["max"] - stats["min"] < 1e-6:
            if stats["std"] > 1e-6:
                normalized = (value - stats["mean"]) / stats["std"]
            else:
                normalized = 0.0
        else:
            # Min-max normalization
            normalized = (value - stats["min"]) / (stats["max"] - stats["min"])
        
        # Clip to [0, 1] range
        normalized = max(0.0, min(1.0, normalized))
        
        return normalized
    
    def update_statistics(self):
        """Update normalization statistics from history."""
        if len(self.inter_class_history) > 0:
            self.inter_class_stats = {
                "min": min(self.inter_class_history),
                "max": max(self.inter_class_history),
                "mean": np.mean(self.inter_class_history),
                "std": np.std(self.inter_class_history)
            }
        
        if len(self.intra_class_history) > 0:
            self.intra_class_stats = {
                "min": min(self.intra_class_history),
                "max": max(self.intra_class_history),
                "mean": np.mean(self.intra_class_history),
                "std": np.std(self.intra_class_history)
            }
        
        if len(self.golden_loss_history) > 0:
            self.golden_loss_stats = {
                "min": min(self.golden_loss_history),
                "max": max(self.golden_loss_history),
                "mean": np.mean(self.golden_loss_history),
                "std": np.std(self.golden_loss_history)
            }
    
    def compute_reward(
        self,
        inter_class_distance: float,
        intra_class_distance: float,
        golden_loss: float,
        update_history: bool = True
    ) -> Dict[str, Any]:
        """
        Compute the reward signal from the components.
        
        Args:
            inter_class_distance: Average inter-class distance
            intra_class_distance: Average intra-class distance
            golden_loss: Golden loss from classifier evaluation
            update_history: Whether to update the history for normalization
        
        Returns:
            Dictionary containing reward components and final reward
        """
        # Update history
        if update_history:
            self.inter_class_history.append(inter_class_distance)
            self.intra_class_history.append(intra_class_distance)
            self.golden_loss_history.append(golden_loss)
            
            # Keep history size manageable
            if len(self.inter_class_history) > self.reward_history_size:
                self.inter_class_history = self.inter_class_history[-self.reward_history_size:]
                self.intra_class_history = self.intra_class_history[-self.reward_history_size:]
                self.golden_loss_history = self.golden_loss_history[-self.reward_history_size:]
            
            # Update statistics
            self.update_statistics()
        
        # Normalize components
        norm_inter_class = self.normalize_component(
            inter_class_distance, "inter_class", self.inter_class_stats
        )
        norm_intra_class = self.normalize_component(
            intra_class_distance, "intra_class", self.intra_class_stats
        )
        norm_golden_loss = self.normalize_component(
            golden_loss, "golden_loss", self.golden_loss_stats
        )
        
        # Compute weighted components
        # Higher inter-class distance is better (positive contribution)
        inter_class_component = self.inter_class_weight * norm_inter_class
        
        # Lower intra-class distance is better (negative contribution)
        intra_class_component = -self.intra_class_weight * norm_intra_class
        
        # Lower golden loss is better (negative contribution)
        golden_loss_component = -self.golden_loss_weight * norm_golden_loss
        
        # Compute final reward
        reward = inter_class_component + intra_class_component + golden_loss_component
        
        # Clip reward if enabled
        if self.clip_rewards:
            reward = max(-self.clip_value, min(self.clip_value, reward))
        
        # Update reward history
        if update_history:
            self.reward_history.append(reward)
            if len(self.reward_history) > self.reward_history_size:
                self.reward_history = self.reward_history[-self.reward_history_size:]
        
        return {
            "reward": reward,
            "inter_class_distance": inter_class_distance,
            "intra_class_distance": intra_class_distance,
            "golden_loss": golden_loss,
            "norm_inter_class": norm_inter_class,
            "norm_intra_class": norm_intra_class,
            "norm_golden_loss": norm_golden_loss,
            "inter_class_component": inter_class_component,
            "intra_class_component": intra_class_component,
            "golden_loss_component": golden_loss_component,
            "weights": {
                "inter_class": self.inter_class_weight,
                "intra_class": self.intra_class_weight,
                "golden_loss": self.golden_loss_weight
            }
        }
    
    def compute_batch_rewards(
        self,
        metrics_batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute rewards for a batch of metrics.
        
        Args:
            metrics_batch: List of metric dictionaries
        
        Returns:
            List of reward dictionaries
        """
        rewards = []
        
        for metrics in metrics_batch:
            reward_info = self.compute_reward(
                inter_class_distance=metrics.get("avg_inter_class_distance", 0.0),
                intra_class_distance=metrics.get("avg_intra_class_distance", 0.0),
                golden_loss=metrics.get("golden_loss", 1.0),
                update_history=True
            )
            rewards.append(reward_info)
        
        return rewards
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reward history."""
        if not self.reward_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        return {
            "mean": float(np.mean(self.reward_history)),
            "std": float(np.std(self.reward_history)),
            "min": float(min(self.reward_history)),
            "max": float(max(self.reward_history)),
            "count": len(self.reward_history)
        }
    
    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all components."""
        return {
            "inter_class": self.inter_class_stats,
            "intra_class": self.intra_class_stats,
            "golden_loss": self.golden_loss_stats,
            "reward": self.get_reward_statistics()
        }
    
    def reset_history(self):
        """Reset the reward history."""
        self.inter_class_history = []
        self.intra_class_history = []
        self.golden_loss_history = []
        self.reward_history = []
        
        # Reset statistics to defaults
        self.inter_class_stats = {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1}
        self.intra_class_stats = {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1}
        self.golden_loss_stats = {"min": 0.0, "max": 2.0, "mean": 1.0, "std": 0.5}
        
        logging.info("Reward history reset")
    
    def save_state(self, filepath: str):
        """Save the reward model state."""
        import json
        
        state = {
            "weights": {
                "inter_class": self.inter_class_weight,
                "intra_class": self.intra_class_weight,
                "golden_loss": self.golden_loss_weight
            },
            "config": {
                "use_min_max_normalization": self.use_min_max_normalization,
                "clip_rewards": self.clip_rewards,
                "clip_value": self.clip_value,
                "reward_history_size": self.reward_history_size
            },
            "statistics": self.get_component_statistics(),
            "history_sizes": {
                "inter_class": len(self.inter_class_history),
                "intra_class": len(self.intra_class_history),
                "golden_loss": len(self.golden_loss_history),
                "reward": len(self.reward_history)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logging.info(f"Reward model state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load the reward model state."""
        import json
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Load weights
        weights = state["weights"]
        self.inter_class_weight = weights["inter_class"]
        self.intra_class_weight = weights["intra_class"]
        self.golden_loss_weight = weights["golden_loss"]
        
        # Load config
        config = state["config"]
        self.use_min_max_normalization = config["use_min_max_normalization"]
        self.clip_rewards = config["clip_rewards"]
        self.clip_value = config["clip_value"]
        self.reward_history_size = config["reward_history_size"]
        
        # Load statistics
        statistics = state["statistics"]
        self.inter_class_stats = statistics["inter_class"]
        self.intra_class_stats = statistics["intra_class"]
        self.golden_loss_stats = statistics["golden_loss"]
        
        logging.info(f"Reward model state loaded from {filepath}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            "weights": {
                "inter_class": self.inter_class_weight,
                "intra_class": self.intra_class_weight,
                "golden_loss": self.golden_loss_weight
            },
            "normalization": {
                "use_min_max": self.use_min_max_normalization,
                "clip_rewards": self.clip_rewards,
                "clip_value": self.clip_value
            },
            "history_size": self.reward_history_size
        }

