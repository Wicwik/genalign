#!/usr/bin/env python3
"""
Main training script for GenAlign RL system.
"""

import argparse
import yaml
import torch
import logging
import os
import sys
from pathlib import Path

from src.data import GoldenDataset, ICLSampler, SyntheticDataset, create_dataloader
from src.generator import LlamaGenerator, ClassificationPromptTemplate
from src.classifier import RoBERTaClassifier, ClassifierTrainer
from src.metrics import DistanceCalculator
from src.reward import RewardModel
from src.rl import PPOTrainerWrapper, SimplifiedPPOTrainer
from src.utils import setup_logging, ExperimentTracker


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_components(config: dict):
    """Setup all components for training."""
    # Setup logging
    logger = setup_logging(
        log_level="INFO",
        use_wandb=config["logging"]["use_wandb"],
        project_name=config["logging"]["project_name"]
    )
    
    # Load golden dataset
    logger.info("Loading golden dataset...")
    golden_dataset = GoldenDataset(
        data_path=config["data"]["golden_data_path"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        max_length=config["data"]["max_length"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"]
    )
    
    # Setup ICL sampler
    icl_sampler = ICLSampler(
        golden_dataset=golden_dataset,
        num_icl_examples=config["generation"]["num_icl_examples"],
        sampling_strategy="balanced"
    )
    
    # Setup generator
    logger.info("Initializing generator...")
    generator = LlamaGenerator(
        model_name=config["models"]["generator"]["name"],
        cache_dir=config["models"]["generator"]["cache_dir"],
        quantization=config["models"]["generator"]["quantization"],
        use_lora=config["models"]["generator"]["use_lora"],
        lora_rank=config["models"]["generator"]["lora_rank"],
        lora_alpha=config["models"]["generator"]["lora_alpha"],
        lora_dropout=config["models"]["generator"]["lora_dropout"]
    )
    
    # Setup prompt template
    prompt_template = ClassificationPromptTemplate(
        task_description="text classification"
    )
    generator.set_prompt_template(prompt_template)
    
    # Setup classifier
    logger.info("Initializing classifier...")
    classifier = RoBERTaClassifier(
        model_name=config["models"]["classifier"]["name"],
        num_labels=golden_dataset.num_labels,
        cache_dir=config["models"]["classifier"]["cache_dir"]
    )
    classifier.set_label_mapping(golden_dataset.label_to_id, golden_dataset.id_to_label)
    
    # Setup distance calculator
    distance_calculator = DistanceCalculator(
        distance_metric="cosine",
        use_centroids=True
    )
    
    # Setup reward model
    reward_model = RewardModel(
        inter_class_weight=config["reward"]["weights"]["inter_class_distance"],
        intra_class_weight=config["reward"]["weights"]["intra_class_distance"],
        golden_loss_weight=config["reward"]["weights"]["golden_loss"],
        use_min_max_normalization=config["reward"]["normalization"]["use_min_max"],
        clip_rewards=config["reward"]["normalization"]["clip_rewards"],
        clip_value=config["reward"]["normalization"]["clip_value"]
    )
    
    # Setup PPO trainer
    ppo_config = {
        "learning_rate": float(config["training"]["learning_rate"]),
        "batch_size": config["training"]["batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "max_grad_norm": config["training"]["max_grad_norm"],
        "ppo_epochs": 4,
        "cliprange": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.1
    }
    
    # Use simplified PPO trainer for now
    ppo_trainer = SimplifiedPPOTrainer(
        generator=generator,
        reward_model=reward_model,
        config=ppo_config
    )
    
    return {
        "logger": logger,
        "golden_dataset": golden_dataset,
        "icl_sampler": icl_sampler,
        "generator": generator,
        "classifier": classifier,
        "distance_calculator": distance_calculator,
        "reward_model": reward_model,
        "ppo_trainer": ppo_trainer
    }


def train_iteration(
    components: dict,
    config: dict,
    iteration: int,
    experiment_tracker: ExperimentTracker
) -> dict:
    """Perform one training iteration."""
    logger = components["logger"]
    golden_dataset = components["golden_dataset"]
    icl_sampler = components["icl_sampler"]
    generator : LlamaGenerator = components["generator"]
    classifier = components["classifier"]
    distance_calculator = components["distance_calculator"]
    reward_model = components["reward_model"]
    ppo_trainer = components["ppo_trainer"]
    
    logger.info(f"Starting iteration {iteration}")
    
    # Step 1: Generate synthetic data
    logger.info("Generating synthetic data...")
    icl_examples = icl_sampler.sample_examples()
    
    generated_samples = generator.generate_samples(
        icl_examples=icl_examples,
        num_samples=config["generation"]["num_samples_per_iteration"],
        temperature=config["generation"]["temperature"],
        target_classes=golden_dataset.unique_labels,
    )
    
    if not generated_samples:
        logger.warning("No samples generated successfully")
        return {"success": False, "error": "No samples generated"}
    
    generated_texts = [sample[0] for sample in generated_samples]
    generated_labels = [sample[1] for sample in generated_samples]
    
    logger.info(f"Generated {len(generated_samples)} samples")
    
    # Step 2: Reload classifier to pre-trained weights
    logger.info("Reloading classifier to pre-trained weights...")
    classifier.load_model(config["models"]["classifier"]["name"])
    classifier.set_label_mapping(golden_dataset.label_to_id, golden_dataset.id_to_label)
    
    # Step 3: Train classifier on synthetic data
    logger.info("Training classifier on synthetic data...")
    
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.add_samples(
        generated_texts, 
        generated_labels,
        [golden_dataset.id_to_label[label] for label in generated_labels],
        iteration
    )
    
    train_dataloader = create_dataloader(
        synthetic_dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    classifier_trainer = ClassifierTrainer(classifier)
    training_history = classifier_trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=train_dataloader,  # Using same data for validation
        num_epochs=config["training"]["num_epochs"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        save_best_model=False
    )
    
    # Step 4: Evaluate classifier on golden data
    logger.info("Evaluating classifier on golden data...")
    golden_texts, golden_labels = golden_dataset.get_val_data()
    
    eval_results = classifier_trainer.evaluate_on_golden_data(
        golden_texts=golden_texts,
        golden_labels=golden_labels
    )
    
    golden_loss = eval_results["loss"]
    golden_accuracy = eval_results["accuracy"]
    
    logger.info(f"Golden loss: {golden_loss:.4f}, Golden accuracy: {golden_accuracy:.4f}")
    
    # Step 5: Compute distance metrics
    logger.info("Computing distance metrics...")
    distance_metrics = distance_calculator.compute_distance_metrics(
        texts=generated_texts,
        labels=generated_labels,
        classifier=classifier
    )
    
    inter_class_distance = distance_metrics["avg_inter_class_distance"]
    intra_class_distance = distance_metrics["avg_intra_class_distance"]
    
    logger.info(f"Inter-class distance: {inter_class_distance:.4f}")
    logger.info(f"Intra-class distance: {intra_class_distance:.4f}")
    
    # Step 6: Compute reward
    reward_info = reward_model.compute_reward(
        inter_class_distance=inter_class_distance,
        intra_class_distance=intra_class_distance,
        golden_loss=golden_loss
    )
    
    reward = reward_info["reward"]
    logger.info(f"Reward: {reward:.4f}")
    
    # Step 7: Update generator with PPO
    logger.info("Updating generator with PPO...")
    
    # Create prompts for PPO training
    prompts = []
    for i, (text, label) in enumerate(zip(generated_texts, generated_labels)):
        prompt = f"Generate a text sample for class {golden_dataset.id_to_label[label]}"
        prompts.append(prompt)
    
    ppo_stats = ppo_trainer.train_step_simplified(
        prompts=prompts,
        generated_texts=generated_texts,
        rewards=[reward] * len(generated_texts)
    )
    
    # Step 8: Log metrics
    metrics = {
        "iteration": iteration,
        "num_generated_samples": len(generated_samples),
        "golden_loss": golden_loss,
        "golden_accuracy": golden_accuracy,
        "inter_class_distance": inter_class_distance,
        "intra_class_distance": intra_class_distance,
        "reward": reward,
        "policy_loss": ppo_stats.get("policy_loss", 0.0),
        "mean_reward": ppo_stats.get("mean_reward", 0.0)
    }
    
    experiment_tracker.log_metrics(metrics, iteration)
    
    # Step 9: Save checkpoint
    if iteration % config["logging"]["save_interval"] == 0:
        checkpoint_info = {
            "iteration": iteration,
            "golden_loss": golden_loss,
            "reward": reward,
            "num_samples": len(generated_samples)
        }
        
        experiment_tracker.save_checkpoint(
            model_state=generator.model.state_dict(),
            step=iteration,
            additional_info=checkpoint_info
        )
    
    return {
        "success": True,
        "metrics": metrics,
        "generated_samples": generated_samples,
        "golden_loss": golden_loss,
        "reward": reward
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GenAlign RL system")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup components
    components = setup_components(config)
    logger = components["logger"]
    
    # Create experiment directory
    from utils.logging import create_experiment_directory
    experiment_dir = create_experiment_directory(
        base_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
    
    # Setup experiment tracker
    experiment_tracker = ExperimentTracker(
        experiment_dir=experiment_dir,
        use_wandb=config["logging"]["use_wandb"],
        project_name=config["logging"]["project_name"]
    )
    
    # Save configuration
    import yaml
    with open(os.path.join(experiment_dir, "configs", "config.yaml"), 'w') as f:
        yaml.dump(config, f, indent=2)
    
    logger.info(f"Starting training with {config['convergence']['max_iterations']} max iterations")
    
    # Training loop
    best_reward = float('-inf')
    patience_counter = 0
    
    for iteration in range(1, config["convergence"]["max_iterations"] + 1):
        try:
            # Perform training iteration
            result = train_iteration(components, config, iteration, experiment_tracker)
            
            if not result["success"]:
                logger.error(f"Iteration {iteration} failed: {result.get('error', 'Unknown error')}")
                continue
            
            # Check convergence
            current_reward = result["reward"]
            
            if current_reward > best_reward + config["convergence"]["min_delta"]:
                best_reward = current_reward
                patience_counter = 0
                logger.info(f"New best reward: {best_reward:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} iterations")
            
            # Early stopping
            if (config["convergence"]["early_stopping"] and 
                patience_counter >= config["convergence"]["patience"]):
                logger.info(f"Early stopping at iteration {iteration}")
                break
            
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {str(e)}")
            continue
    
    logger.info("Training completed")
    
    # Save final model
    final_checkpoint_path = os.path.join(experiment_dir, "final_model")
    components["generator"].save_model(final_checkpoint_path)
    
    # Close experiment tracker
    experiment_tracker.close()
    
    logger.info(f"Experiment completed. Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()

