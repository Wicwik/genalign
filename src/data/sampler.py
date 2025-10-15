import random
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch


class ICLSampler:
    """In-Context Learning example sampler for selecting diverse examples from golden data."""
    
    def __init__(
        self,
        golden_dataset,
        num_icl_examples: int = 3,
        sampling_strategy: str = "random",  # "random", "diverse", "balanced"
        random_state: int = 42
    ):
        """
        Initialize the ICL sampler.
        
        Args:
            golden_dataset: GoldenDataset instance
            num_icl_examples: Number of examples to select for ICL
            sampling_strategy: Strategy for sampling examples
            random_state: Random seed for reproducibility
        """
        self.golden_dataset = golden_dataset
        self.num_icl_examples = num_icl_examples
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Prepare data for sampling
        self._prepare_sampling_data()
    
    def _prepare_sampling_data(self):
        """Prepare data structures for efficient sampling."""
        self.class_examples = {}
        for class_id in range(self.golden_dataset.num_labels):
            examples = self.golden_dataset.get_class_examples(class_id)
            self.class_examples[class_id] = examples
        
        # For diverse sampling, compute TF-IDF features
        if self.sampling_strategy == "diverse":
            all_texts = [example[0] for examples in self.class_examples.values() for example in examples]
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.tfidf_features = self.tfidf_vectorizer.fit_transform(all_texts)
    
    def sample_examples(self, target_class: int = None) -> List[Tuple[str, int, str]]:
        """
        Sample examples for in-context learning.
        
        Args:
            target_class: Target class for generation (None for balanced sampling)
        
        Returns:
            List of (text, label_id, label_name) tuples
        """
        if self.sampling_strategy == "random":
            return self._random_sampling(target_class)
        elif self.sampling_strategy == "diverse":
            return self._diverse_sampling(target_class)
        elif self.sampling_strategy == "balanced":
            return self._balanced_sampling(target_class)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def _random_sampling(self, target_class: int = None) -> List[Tuple[str, int, str]]:
        """Random sampling strategy."""
        if target_class is not None:
            # Sample from specific class
            examples = self.class_examples[target_class]
            selected = random.sample(examples, min(self.num_icl_examples, len(examples)))
        else:
            # Sample from all classes
            all_examples = []
            for examples in self.class_examples.values():
                all_examples.extend(examples)
            selected = random.sample(all_examples, min(self.num_icl_examples, len(all_examples)))
        
        return [(text, label_id, self.golden_dataset.id_to_label[label_id]) 
                for text, label_id in selected]
    
    def _diverse_sampling(self, target_class: int = None) -> List[Tuple[str, int, str]]:
        """Diverse sampling strategy using TF-IDF similarity."""
        if target_class is not None:
            # Sample diverse examples from specific class
            examples = self.class_examples[target_class]
            if len(examples) <= self.num_icl_examples:
                return [(text, label_id, self.golden_dataset.id_to_label[label_id]) 
                        for text, label_id in examples]
            
            # Get TF-IDF features for this class
            class_texts = [example[0] for example in examples]
            class_features = self.tfidf_vectorizer.transform(class_texts)
            
            # Select diverse examples using cosine similarity
            selected_indices = self._select_diverse_indices(class_features, self.num_icl_examples)
            selected = [examples[i] for i in selected_indices]
        else:
            # Sample diverse examples from all classes
            all_examples = []
            for examples in self.class_examples.values():
                all_examples.extend(examples)
            
            if len(all_examples) <= self.num_icl_examples:
                return [(text, label_id, self.golden_dataset.id_to_label[label_id]) 
                        for text, label_id in all_examples]
            
            # Get TF-IDF features for all examples
            all_texts = [example[0] for example in all_examples]
            all_features = self.tfidf_vectorizer.transform(all_texts)
            
            # Select diverse examples
            selected_indices = self._select_diverse_indices(all_features, self.num_icl_examples)
            selected = [all_examples[i] for i in selected_indices]
        
        return [(text, label_id, self.golden_dataset.id_to_label[label_id]) 
                for text, label_id in selected]
    
    def _balanced_sampling(self, target_class: int = None) -> List[Tuple[str, int, str]]:
        """Balanced sampling strategy ensuring representation from all classes."""
        if target_class is not None:
            # For specific class, use diverse sampling
            return self._diverse_sampling(target_class)
        
        # Sample from all classes with balanced representation
        examples_per_class = max(1, self.num_icl_examples // self.golden_dataset.num_labels)
        selected = []
        
        for class_id in range(self.golden_dataset.num_labels):
            examples = self.class_examples[class_id]
            if examples:
                # Sample examples from this class
                num_to_sample = min(examples_per_class, len(examples))
                class_selected = random.sample(examples, num_to_sample)
                selected.extend(class_selected)
        
        # If we need more examples, fill with random sampling
        if len(selected) < self.num_icl_examples:
            all_examples = []
            for examples in self.class_examples.values():
                all_examples.extend(examples)
            remaining = self.num_icl_examples - len(selected)
            additional = random.sample(all_examples, min(remaining, len(all_examples)))
            selected.extend(additional)
        
        # Shuffle and limit to requested number
        random.shuffle(selected)
        selected = selected[:self.num_icl_examples]
        
        return [(text, label_id, self.golden_dataset.id_to_label[label_id]) 
                for text, label_id in selected]
    
    def _select_diverse_indices(self, features, num_to_select: int) -> List[int]:
        """
        Select diverse indices using cosine similarity.
        
        Args:
            features: TF-IDF features matrix
            num_to_select: Number of indices to select
        
        Returns:
            List of selected indices
        """
        if features.shape[0] <= num_to_select:
            return list(range(features.shape[0]))
        
        # Start with a random example
        selected_indices = [random.randint(0, features.shape[0] - 1)]
        
        # Iteratively select the most diverse example
        for _ in range(num_to_select - 1):
            similarities = []
            for i in range(features.shape[0]):
                if i not in selected_indices:
                    # Compute average similarity to already selected examples
                    sims = cosine_similarity(features[i:i+1], features[selected_indices])
                    avg_sim = np.mean(sims)
                    similarities.append((i, avg_sim))
            
            # Select the example with minimum average similarity (most diverse)
            if similarities:
                similarities.sort(key=lambda x: x[1])
                selected_indices.append(similarities[0][0])
        
        return selected_indices
    
    def update_sampling_strategy(self, strategy: str):
        """Update the sampling strategy."""
        self.sampling_strategy = strategy
        if strategy == "diverse":
            self._prepare_sampling_data()
    
    def get_class_statistics(self) -> Dict[int, int]:
        """Get statistics about available examples per class."""
        return {class_id: len(examples) for class_id, examples in self.class_examples.items()}

