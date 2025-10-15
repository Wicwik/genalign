# GenAlign: RL-based Synthetic Data Generation

A reinforcement learning system that improves synthetic data generation by training a generator (Llama-3.1-8B) using PPO, guided by classifier performance on real data and distributional quality metrics.

## Overview

This project implements a closed-loop system where:
1. A generator produces synthetic data using in-context learning
2. A classifier is trained on the synthetic data
3. The classifier's performance on real data, combined with distributional quality metrics, forms a reward signal
4. The generator is optimized using PPO to maximize this reward

## Methodology

The system follows this pipeline:
1. Generate synthetic data using the generator with ICL examples
2. Train a RoBERTa classifier on the synthetic data
3. Evaluate the classifier on golden (real) data to get the golden loss
4. Compute inter-class and intra-class distances of the generated data
5. Combine these metrics into a reward signal
6. Use PPO to align the generator based on the reward
7. Repeat until convergence

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py --config config/config.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --config config/config.yaml --checkpoint outputs/checkpoint_epoch_10
```

## Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Training hyperparameters
- Reward weights
- Convergence criteria
- Data paths

## Project Structure

```
genalign/
├── config/           # Configuration files
├── src/             # Source code modules
│   ├── data/        # Data loading and sampling
│   ├── generator/   # Llama-3.1-8B generator
│   ├── classifier/  # RoBERTa classifier
│   ├── metrics/     # Distance computation
│   ├── reward/      # Reward computation
│   ├── rl/          # PPO training
│   └── utils/       # Utilities
├── scripts/         # Training and evaluation scripts
└── outputs/         # Model checkpoints and logs
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space for model cache

## License

MIT License

