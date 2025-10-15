import logging
import os
import json
import torch
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_wandb: bool = False,
    project_name: str = "genalign-rl",
    run_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for console only)
        use_wandb: Whether to use Weights & Biases
        project_name: W&B project name
        run_name: W&B run name
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("genalign")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Initialize W&B if requested
    if use_wandb and WANDB_AVAILABLE:
        if run_name is None:
            run_name = f"genalign-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        wandb.init(
            project=project_name,
            name=run_name,
            config={}
        )
        logger.info(f"Initialized W&B run: {run_name}")
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("W&B requested but not available. Install with: pip install wandb")
    
    logger.info("Logging setup completed")
    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    use_wandb: bool = False,
    logger: Optional[logging.Logger] = None
):
    """
    Log metrics to console, file, and optionally W&B.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Step number for logging
        use_wandb: Whether to log to W&B
        logger: Logger instance
    """
    # Log to console/file
    if logger:
        log_message = f"Step {step}: " if step is not None else ""
        log_message += ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in metrics.items()])
        logger.info(log_message)
    
    # Log to W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if step is not None:
            wandb_metrics["step"] = step
        wandb.log(wandb_metrics)


def save_experiment_config(
    config: Dict[str, Any],
    save_path: str
):
    """
    Save experiment configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Experiment config saved to {save_path}")


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from file.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"Experiment config loaded from {config_path}")
    return config


def create_experiment_directory(
    base_dir: str,
    experiment_name: str = None
) -> str:
    """
    Create a directory for the experiment.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
    
    Returns:
        Path to the created experiment directory
    """
    if experiment_name is None:
        experiment_name = f"experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ["checkpoints", "logs", "outputs", "configs"]
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    logging.info(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


class ExperimentTracker:
    """Simple experiment tracker for logging and saving results."""
    
    def __init__(
        self,
        experiment_dir: str,
        use_wandb: bool = False,
        project_name: str = "genalign-rl"
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_dir: Directory for the experiment
            use_wandb: Whether to use W&B
            project_name: W&B project name
        """
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb
        self.project_name = project_name
        
        # Create metrics file
        self.metrics_file = os.path.join(experiment_dir, "metrics.json")
        self.metrics_history = []
        
        # Initialize W&B if requested
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=project_name,
                dir=experiment_dir,
                config={}
            )
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics for a step.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number
        """
        # Add timestamp and step
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "metrics": metrics
        }
        
        # Add to history
        self.metrics_history.append(log_entry)
        
        # Save to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Log to W&B
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
    
    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        step: int,
        additional_info: Dict[str, Any] = None
    ):
        """
        Save a model checkpoint.
        
        Args:
            model_state: Model state dictionary
            step: Step number
            additional_info: Additional information to save
        """
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
        
        checkpoint_data = {
            "step": step,
            "timestamp": time.time(),
            "model_state": model_state,
            "additional_info": additional_info or {}
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, step: int) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            step: Step number
        
        Returns:
            Checkpoint data
        """
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path)
        logging.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint_data
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the metrics history."""
        return self.metrics_history.copy()
    
    def close(self):
        """Close the experiment tracker."""
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
