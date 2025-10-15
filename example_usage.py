#!/usr/bin/env python3
"""
Example usage script for GenAlign RL system.
This script demonstrates how to use the system with the sample data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import GoldenDataset, ICLSampler
from generator import LlamaGenerator, ClassificationPromptTemplate
from classifier import RoBERTaClassifier, ClassifierTrainer
from metrics import DistanceCalculator
from reward import RewardModel
from utils import setup_logging


def main():
    """Example usage of the GenAlign system."""
    
    # Setup logging
    logger = setup_logging(log_level="INFO", use_wandb=False)
    
    logger.info("GenAlign RL System - Example Usage")
    logger.info("=" * 50)
    
    # 1. Load golden dataset
    logger.info("1. Loading golden dataset...")
    golden_dataset = GoldenDataset(
        data_path="./data/sample_data.csv",
        text_column="text",
        label_column="label",
        max_length=512
    )
    
    logger.info(f"Loaded {len(golden_dataset)} samples with {golden_dataset.num_labels} classes")
    logger.info(f"Classes: {list(golden_dataset.label_to_id.keys())}")
    
    # 2. Setup ICL sampler
    logger.info("2. Setting up ICL sampler...")
    icl_sampler = ICLSampler(
        golden_dataset=golden_dataset,
        num_icl_examples=3,
        sampling_strategy="diverse"
    )
    
    # Sample some examples
    icl_examples = icl_sampler.sample_examples()
    logger.info(f"Sampled {len(icl_examples)} ICL examples:")
    for i, (text, label_id, label_name) in enumerate(icl_examples):
        logger.info(f"  {i+1}. [{label_name}] {text[:50]}...")
    
    # 3. Setup generator (using a smaller model for demo)
    logger.info("3. Setting up generator...")
    try:
        generator = LlamaGenerator(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            cache_dir="./cache",
            quantization="4bit",
            use_lora=True
        )
        
        # Setup prompt template
        prompt_template = ClassificationPromptTemplate(
            task_description="sentiment analysis"
        )
        generator.set_prompt_template(prompt_template)
        
        logger.info("Generator setup completed")
        
    except Exception as e:
        logger.warning(f"Could not load Llama model: {e}")
        logger.info("Skipping generator demo due to model loading issues")
        generator = None
    
    # 4. Setup classifier
    logger.info("4. Setting up classifier...")
    classifier = RoBERTaClassifier(
        model_name="roberta-base",
        num_labels=golden_dataset.num_labels,
        cache_dir="./cache"
    )
    classifier.set_label_mapping(golden_dataset.label_to_id, golden_dataset.id_to_label)
    
    logger.info("Classifier setup completed")
    
    # 5. Setup distance calculator
    logger.info("5. Setting up distance calculator...")
    distance_calculator = DistanceCalculator(
        distance_metric="cosine",
        use_centroids=True
    )
    
    # 6. Setup reward model
    logger.info("6. Setting up reward model...")
    reward_model = RewardModel(
        inter_class_weight=1.0,
        intra_class_weight=1.0,
        golden_loss_weight=1.0
    )
    
    # 7. Demo: Generate some samples (if generator is available)
    if generator is not None:
        logger.info("7. Generating synthetic samples...")
        try:
            generated_samples = generator.generate_samples(
                icl_examples=icl_examples,
                num_samples=5,
                temperature=0.7
            )
            
            if generated_samples:
                logger.info(f"Generated {len(generated_samples)} samples:")
                for i, (text, label_id, label_name) in enumerate(generated_samples):
                    logger.info(f"  {i+1}. [{label_name}] {text[:100]}...")
            else:
                logger.warning("No samples were generated successfully")
                
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
    
    # 8. Demo: Train classifier on golden data
    logger.info("8. Training classifier on golden data...")
    from data.dataset import create_dataloader
    
    # Create dataloader for golden data
    train_texts, train_labels = golden_dataset.get_train_data()
    val_texts, val_labels = golden_dataset.get_val_data()
    
    # Create datasets
    from data.dataset import SyntheticDataset
    train_dataset = SyntheticDataset()
    train_dataset.add_samples(
        train_texts, 
        train_labels,
        [golden_dataset.id_to_label[label] for label in train_labels],
        0
    )
    
    val_dataset = SyntheticDataset()
    val_dataset.add_samples(
        val_texts, 
        val_labels,
        [golden_dataset.id_to_label[label] for label in val_labels],
        0
    )
    
    train_dataloader = create_dataloader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, batch_size=4, shuffle=False)
    
    # Train classifier
    classifier_trainer = ClassifierTrainer(classifier)
    training_history = classifier_trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=2,
        save_best_model=False
    )
    
    logger.info("Classifier training completed")
    
    # 9. Demo: Evaluate classifier
    logger.info("9. Evaluating classifier...")
    eval_results = classifier_trainer.evaluate_on_golden_data(
        golden_texts=val_texts,
        golden_labels=val_labels
    )
    
    logger.info(f"Evaluation results:")
    logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
    logger.info(f"  Loss: {eval_results['loss']:.4f}")
    
    # 10. Demo: Compute distance metrics (using golden data as example)
    logger.info("10. Computing distance metrics...")
    distance_metrics = distance_calculator.compute_distance_metrics(
        texts=train_texts[:20],  # Use subset for demo
        labels=train_labels[:20],
        classifier=classifier
    )
    
    logger.info(f"Distance metrics:")
    logger.info(f"  Inter-class distance: {distance_metrics['avg_inter_class_distance']:.4f}")
    logger.info(f"  Intra-class distance: {distance_metrics['avg_intra_class_distance']:.4f}")
    logger.info(f"  Quality score: {distance_metrics['quality_score']:.4f}")
    
    # 11. Demo: Compute reward
    logger.info("11. Computing reward...")
    reward_info = reward_model.compute_reward(
        inter_class_distance=distance_metrics['avg_inter_class_distance'],
        intra_class_distance=distance_metrics['avg_intra_class_distance'],
        golden_loss=eval_results['loss']
    )
    
    logger.info(f"Reward computation:")
    logger.info(f"  Final reward: {reward_info['reward']:.4f}")
    logger.info(f"  Inter-class component: {reward_info['inter_class_component']:.4f}")
    logger.info(f"  Intra-class component: {reward_info['intra_class_component']:.4f}")
    logger.info(f"  Golden loss component: {reward_info['golden_loss_component']:.4f}")
    
    logger.info("=" * 50)
    logger.info("Example usage completed successfully!")
    logger.info("To run the full training pipeline, use:")
    logger.info("  python scripts/train.py --config config/config.yaml")
    logger.info("To evaluate a trained model, use:")
    logger.info("  python scripts/evaluate.py --config config/config.yaml --checkpoint <path_to_checkpoint>")


if __name__ == "__main__":
    main()

