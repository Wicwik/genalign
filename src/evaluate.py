#!/usr/bin/env python3
"""
Evaluation script for GenAlign RL system.
"""

import argparse
import yaml
import torch
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from genalign.data import GoldenDataset, ICLSampler
from genalign.generator import Generator, ClassificationPromptTemplate
from genalign.classifier import Classifier, ClassifierTrainer
from genalign.metrics import DistanceCalculator
from genalign.reward import RewardModel
from genalign.utils import setup_logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_components(config: dict):
    """Setup all components for evaluation."""
    # Setup logging
    logger = setup_logging(log_level="INFO", use_wandb=False)
    
    # Load golden dataset
    logger.info("Loading golden dataset...")
    golden_dataset = GoldenDataset(
        data_path=config["data"]["golden_data_path"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        max_length=config["data"]["max_length"]
    )
    
    # Setup ICL sampler
    icl_sampler = ICLSampler(
        golden_dataset=golden_dataset,
        num_icl_examples=config["generation"]["num_icl_examples"],
        sampling_strategy="diverse"
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
    classifier = Classifier(
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
    
    return {
        "logger": logger,
        "golden_dataset": golden_dataset,
        "icl_sampler": icl_sampler,
        "generator": generator,
        "classifier": classifier,
        "distance_calculator": distance_calculator,
        "reward_model": reward_model
    }


def evaluate_baseline(components: dict, config: dict) -> dict:
    """Evaluate the baseline (pre-trained) generator."""
    logger = components["logger"]
    golden_dataset = components["golden_dataset"]
    icl_sampler = components["icl_sampler"]
    generator = components["generator"]
    classifier = components["classifier"]
    distance_calculator = components["distance_calculator"]
    reward_model = components["reward_model"]
    
    logger.info("Evaluating baseline generator...")
    
    # Generate samples with baseline generator
    icl_examples = icl_sampler.sample_examples()
    
    generated_samples = generator.generate_samples(
        icl_examples=icl_examples,
        num_samples=config["generation"]["num_samples_per_iteration"],
        temperature=config["generation"]["temperature"]
    )
    
    if not generated_samples:
        logger.warning("No samples generated with baseline generator")
        return {"success": False}
    
    generated_texts = [sample[0] for sample in generated_samples]
    generated_labels = [sample[1] for sample in generated_samples]
    
    # Train classifier on generated data
    from data.dataset import SyntheticDataset, create_dataloader
    
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.add_samples(
        generated_texts, 
        generated_labels,
        [golden_dataset.id_to_label[label] for label in generated_labels],
        0
    )
    
    train_dataloader = create_dataloader(
        synthetic_dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    classifier_trainer = ClassifierTrainer(classifier)
    training_history = classifier_trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=train_dataloader,
        num_epochs=config["training"]["num_epochs"],
        save_best_model=False
    )
    
    # Evaluate on golden data
    golden_texts, golden_labels = golden_dataset.get_val_data()
    
    eval_results = classifier_trainer.evaluate_on_golden_data(
        golden_texts=golden_texts,
        golden_labels=golden_labels
    )
    
    # Compute distance metrics
    distance_metrics = distance_calculator.compute_distance_metrics(
        texts=generated_texts,
        labels=generated_labels,
        classifier=classifier
    )
    
    # Compute reward
    reward_info = reward_model.compute_reward(
        inter_class_distance=distance_metrics["avg_inter_class_distance"],
        intra_class_distance=distance_metrics["avg_intra_class_distance"],
        golden_loss=eval_results["loss"]
    )
    
    return {
        "success": True,
        "generated_samples": generated_samples,
        "golden_loss": eval_results["loss"],
        "golden_accuracy": eval_results["accuracy"],
        "inter_class_distance": distance_metrics["avg_inter_class_distance"],
        "intra_class_distance": distance_metrics["avg_intra_class_distance"],
        "reward": reward_info["reward"],
        "distance_metrics": distance_metrics,
        "eval_results": eval_results
    }


def evaluate_trained_model(components: dict, config: dict, checkpoint_path: str) -> dict:
    """Evaluate the trained generator."""
    logger = components["logger"]
    golden_dataset = components["golden_dataset"]
    icl_sampler = components["icl_sampler"]
    generator = components["generator"]
    classifier = components["classifier"]
    distance_calculator = components["distance_calculator"]
    reward_model = components["reward_model"]
    
    logger.info(f"Evaluating trained generator from {checkpoint_path}...")
    
    # Load trained model
    generator.load_model(checkpoint_path)
    
    # Generate samples with trained generator
    icl_examples = icl_sampler.sample_examples()
    
    generated_samples = generator.generate_samples(
        icl_examples=icl_examples,
        num_samples=config["generation"]["num_samples_per_iteration"],
        temperature=config["generation"]["temperature"]
    )
    
    if not generated_samples:
        logger.warning("No samples generated with trained generator")
        return {"success": False}
    
    generated_texts = [sample[0] for sample in generated_samples]
    generated_labels = [sample[1] for sample in generated_samples]
    
    # Train classifier on generated data
    from data.dataset import SyntheticDataset, create_dataloader
    
    synthetic_dataset = SyntheticDataset()
    synthetic_dataset.add_samples(
        generated_texts, 
        generated_labels,
        [golden_dataset.id_to_label[label] for label in generated_labels],
        0
    )
    
    train_dataloader = create_dataloader(
        synthetic_dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    classifier_trainer = ClassifierTrainer(classifier)
    training_history = classifier_trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=train_dataloader,
        num_epochs=config["training"]["num_epochs"],
        save_best_model=False
    )
    
    # Evaluate on golden data
    golden_texts, golden_labels = golden_dataset.get_val_data()
    
    eval_results = classifier_trainer.evaluate_on_golden_data(
        golden_texts=golden_texts,
        golden_labels=golden_labels
    )
    
    # Compute distance metrics
    distance_metrics = distance_calculator.compute_distance_metrics(
        texts=generated_texts,
        labels=generated_labels,
        classifier=classifier
    )
    
    # Compute reward
    reward_info = reward_model.compute_reward(
        inter_class_distance=distance_metrics["avg_inter_class_distance"],
        intra_class_distance=distance_metrics["avg_intra_class_distance"],
        golden_loss=eval_results["loss"]
    )
    
    return {
        "success": True,
        "generated_samples": generated_samples,
        "golden_loss": eval_results["loss"],
        "golden_accuracy": eval_results["accuracy"],
        "inter_class_distance": distance_metrics["avg_inter_class_distance"],
        "intra_class_distance": distance_metrics["avg_intra_class_distance"],
        "reward": reward_info["reward"],
        "distance_metrics": distance_metrics,
        "eval_results": eval_results
    }


def create_evaluation_report(
    baseline_results: dict,
    trained_results: dict,
    output_dir: str
):
    """Create a comprehensive evaluation report."""
    logger = logging.getLogger("genalign")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison table
    comparison_data = {
        "Metric": [
            "Golden Loss",
            "Golden Accuracy",
            "Inter-class Distance",
            "Intra-class Distance",
            "Reward",
            "Quality Score"
        ],
        "Baseline": [
            f"{baseline_results['golden_loss']:.4f}",
            f"{baseline_results['golden_accuracy']:.4f}",
            f"{baseline_results['inter_class_distance']:.4f}",
            f"{baseline_results['intra_class_distance']:.4f}",
            f"{baseline_results['reward']:.4f}",
            f"{baseline_results['distance_metrics']['quality_score']:.4f}"
        ],
        "Trained": [
            f"{trained_results['golden_loss']:.4f}",
            f"{trained_results['golden_accuracy']:.4f}",
            f"{trained_results['inter_class_distance']:.4f}",
            f"{trained_results['intra_class_distance']:.4f}",
            f"{trained_results['reward']:.4f}",
            f"{trained_results['distance_metrics']['quality_score']:.4f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, "comparison.csv"), index=False)
    
    # Create improvement analysis
    improvements = {
        "Golden Loss": (trained_results['golden_loss'] - baseline_results['golden_loss']) / baseline_results['golden_loss'] * 100,
        "Golden Accuracy": (trained_results['golden_accuracy'] - baseline_results['golden_accuracy']) / baseline_results['golden_accuracy'] * 100,
        "Inter-class Distance": (trained_results['inter_class_distance'] - baseline_results['inter_class_distance']) / baseline_results['inter_class_distance'] * 100,
        "Intra-class Distance": (trained_results['intra_class_distance'] - baseline_results['intra_class_distance']) / baseline_results['intra_class_distance'] * 100,
        "Reward": (trained_results['reward'] - baseline_results['reward']) / abs(baseline_results['reward']) * 100 if baseline_results['reward'] != 0 else 0,
        "Quality Score": (trained_results['distance_metrics']['quality_score'] - baseline_results['distance_metrics']['quality_score']) / baseline_results['distance_metrics']['quality_score'] * 100
    }
    
    improvement_df = pd.DataFrame(list(improvements.items()), columns=["Metric", "Improvement (%)"])
    improvement_df.to_csv(os.path.join(output_dir, "improvements.csv"), index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Metrics comparison
    metrics = ["Golden Loss", "Golden Accuracy", "Inter-class Distance", "Intra-class Distance", "Reward"]
    baseline_values = [
        baseline_results['golden_loss'],
        baseline_results['golden_accuracy'],
        baseline_results['inter_class_distance'],
        baseline_results['intra_class_distance'],
        baseline_results['reward']
    ]
    trained_values = [
        trained_results['golden_loss'],
        trained_results['golden_accuracy'],
        trained_results['inter_class_distance'],
        trained_results['intra_class_distance'],
        trained_results['reward']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.subplot(2, 2, 1)
    plt.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    plt.bar(x + width/2, trained_values, width, label='Trained', alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Metrics Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Improvement percentages
    plt.subplot(2, 2, 2)
    improvement_values = [improvements[metric] for metric in metrics]
    colors = ['green' if val > 0 else 'red' for val in improvement_values]
    plt.bar(metrics, improvement_values, color=colors, alpha=0.7)
    plt.xlabel('Metrics')
    plt.ylabel('Improvement (%)')
    plt.title('Improvement Over Baseline')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    # Confusion matrices
    plt.subplot(2, 2, 3)
    baseline_cm = np.array(baseline_results['eval_results']['confusion_matrix'])
    sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Baseline Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(2, 2, 4)
    trained_cm = np.array(trained_results['eval_results']['confusion_matrix'])
    sns.heatmap(trained_cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Trained Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_report.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save sample outputs
    baseline_samples_df = pd.DataFrame(baseline_results['generated_samples'], 
                                     columns=['text', 'label_id', 'label_name'])
    baseline_samples_df.to_csv(os.path.join(output_dir, "baseline_samples.csv"), index=False)
    
    trained_samples_df = pd.DataFrame(trained_results['generated_samples'], 
                                    columns=['text', 'label_id', 'label_name'])
    trained_samples_df.to_csv(os.path.join(output_dir, "trained_samples.csv"), index=False)
    
    logger.info(f"Evaluation report saved to {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate GenAlign RL system")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./evaluation_outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup components
    components = setup_components(config)
    logger = components["logger"]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting evaluation...")
    
    # Evaluate baseline
    logger.info("Evaluating baseline generator...")
    baseline_results = evaluate_baseline(components, config)
    
    if not baseline_results["success"]:
        logger.error("Baseline evaluation failed")
        return
    
    # Evaluate trained model
    logger.info("Evaluating trained generator...")
    trained_results = evaluate_trained_model(components, config, args.checkpoint)
    
    if not trained_results["success"]:
        logger.error("Trained model evaluation failed")
        return
    
    # Create evaluation report
    logger.info("Creating evaluation report...")
    create_evaluation_report(baseline_results, trained_results, args.output_dir)
    
    # Print summary
    logger.info("Evaluation Summary:")
    logger.info(f"Baseline - Golden Loss: {baseline_results['golden_loss']:.4f}, "
                f"Golden Accuracy: {baseline_results['golden_accuracy']:.4f}, "
                f"Reward: {baseline_results['reward']:.4f}")
    logger.info(f"Trained - Golden Loss: {trained_results['golden_loss']:.4f}, "
                f"Golden Accuracy: {trained_results['golden_accuracy']:.4f}, "
                f"Reward: {trained_results['reward']:.4f}")
    
    improvement = (trained_results['reward'] - baseline_results['reward']) / abs(baseline_results['reward']) * 100
    logger.info(f"Reward Improvement: {improvement:.2f}%")
    
    logger.info(f"Evaluation completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

