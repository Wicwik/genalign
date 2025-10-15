import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.cluster import KMeans
import logging


class DistanceCalculator:
    """Calculator for inter-class and intra-class distances in generated data."""
    
    def __init__(
        self,
        distance_metric: str = "cosine",  # "cosine", "euclidean"
        use_centroids: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the distance calculator.
        
        Args:
            distance_metric: Distance metric to use ("cosine" or "euclidean")
            use_centroids: Whether to use centroids for inter-class distance
            random_state: Random seed for reproducibility
        """
        self.distance_metric = distance_metric
        self.use_centroids = use_centroids
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
    
    def compute_embeddings(
        self, 
        texts: List[str], 
        classifier,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute embeddings for texts using the classifier.
        
        Args:
            texts: List of text strings
            classifier: RoBERTaClassifier instance
            batch_size: Batch size for processing
        
        Returns:
            Tensor containing embeddings
        """
        return classifier.get_embeddings(texts, batch_size)
    
    def compute_intra_class_distances(
        self, 
        embeddings: torch.Tensor, 
        labels: List[int]
    ) -> Dict[int, float]:
        """
        Compute intra-class distances (average pairwise distance within each class).
        
        Args:
            embeddings: Tensor of embeddings
            labels: List of corresponding labels
        
        Returns:
            Dictionary mapping class_id to average intra-class distance
        """
        embeddings_np = embeddings.cpu().numpy()
        unique_labels = list(set(labels))
        intra_class_distances = {}
        
        for class_id in unique_labels:
            # Get embeddings for this class
            class_indices = [i for i, label in enumerate(labels) if label == class_id]
            if len(class_indices) < 2:
                # Not enough samples for pairwise distance
                intra_class_distances[class_id] = 0.0
                continue
            
            class_embeddings = embeddings_np[class_indices]
            
            # Compute pairwise distances
            if self.distance_metric == "cosine":
                distances = cosine_distances(class_embeddings)
            elif self.distance_metric == "euclidean":
                distances = euclidean_distances(class_embeddings)
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
            # Get upper triangle (excluding diagonal) to avoid double counting
            upper_triangle = np.triu(distances, k=1)
            valid_distances = upper_triangle[upper_triangle > 0]
            
            if len(valid_distances) > 0:
                avg_distance = np.mean(valid_distances)
            else:
                avg_distance = 0.0
            
            intra_class_distances[class_id] = float(avg_distance)
        
        return intra_class_distances
    
    def compute_inter_class_distances(
        self, 
        embeddings: torch.Tensor, 
        labels: List[int]
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute inter-class distances (average distance between different classes).
        
        Args:
            embeddings: Tensor of embeddings
            labels: List of corresponding labels
        
        Returns:
            Dictionary mapping (class1, class2) to average inter-class distance
        """
        embeddings_np = embeddings.cpu().numpy()
        unique_labels = sorted(list(set(labels)))
        inter_class_distances = {}
        
        if self.use_centroids:
            # Use centroids for inter-class distance
            centroids = {}
            for class_id in unique_labels:
                class_indices = [i for i, label in enumerate(labels) if label == class_id]
                if len(class_indices) > 0:
                    class_embeddings = embeddings_np[class_indices]
                    centroids[class_id] = np.mean(class_embeddings, axis=0)
            
            # Compute distances between centroids
            for i, class1 in enumerate(unique_labels):
                for j, class2 in enumerate(unique_labels):
                    if i < j and class1 in centroids and class2 in centroids:
                        if self.distance_metric == "cosine":
                            distance = cosine_distances(
                                centroids[class1].reshape(1, -1),
                                centroids[class2].reshape(1, -1)
                            )[0, 0]
                        elif self.distance_metric == "euclidean":
                            distance = euclidean_distances(
                                centroids[class1].reshape(1, -1),
                                centroids[class2].reshape(1, -1)
                            )[0, 0]
                        else:
                            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
                        
                        inter_class_distances[(class1, class2)] = float(distance)
        else:
            # Use all pairwise distances between classes
            for i, class1 in enumerate(unique_labels):
                for j, class2 in enumerate(unique_labels):
                    if i < j:
                        class1_indices = [idx for idx, label in enumerate(labels) if label == class1]
                        class2_indices = [idx for idx, label in enumerate(labels) if label == class2]
                        
                        if len(class1_indices) > 0 and len(class2_indices) > 0:
                            class1_embeddings = embeddings_np[class1_indices]
                            class2_embeddings = embeddings_np[class2_indices]
                            
                            # Compute all pairwise distances between classes
                            if self.distance_metric == "cosine":
                                distances = cosine_distances(class1_embeddings, class2_embeddings)
                            elif self.distance_metric == "euclidean":
                                distances = euclidean_distances(class1_embeddings, class2_embeddings)
                            else:
                                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
                            
                            avg_distance = np.mean(distances)
                            inter_class_distances[(class1, class2)] = float(avg_distance)
        
        return inter_class_distances
    
    def compute_distance_metrics(
        self, 
        texts: List[str], 
        labels: List[int],
        classifier,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Compute all distance metrics for the given data.
        
        Args:
            texts: List of text strings
            labels: List of corresponding labels
            classifier: RoBERTaClassifier instance
            batch_size: Batch size for processing
        
        Returns:
            Dictionary containing all distance metrics
        """
        # Compute embeddings
        embeddings = self.compute_embeddings(texts, classifier, batch_size)
        
        # Compute intra-class distances
        intra_class_distances = self.compute_intra_class_distances(embeddings, labels)
        
        # Compute inter-class distances
        inter_class_distances = self.compute_inter_class_distances(embeddings, labels)
        
        # Compute summary statistics
        avg_intra_class_distance = np.mean(list(intra_class_distances.values()))
        avg_inter_class_distance = np.mean(list(inter_class_distances.values()))
        
        # Compute class balance
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_balance = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        # Compute quality score
        quality_score = self._compute_quality_score(
            avg_inter_class_distance, avg_intra_class_distance, 0.0, 0.0
        )
        
        return {
            "intra_class_distances": intra_class_distances,
            "inter_class_distances": inter_class_distances,
            "avg_intra_class_distance": float(avg_intra_class_distance),
            "avg_inter_class_distance": float(avg_inter_class_distance),
            "class_balance": class_balance,
            "num_classes": len(unique_labels),
            "total_samples": len(texts),
            "distance_metric": self.distance_metric,
            "use_centroids": self.use_centroids,
            "quality_score": quality_score
        }
    
    def compute_separation_ratio(
        self, 
        avg_inter_class_distance: float, 
        avg_intra_class_distance: float
    ) -> float:
        """
        Compute the separation ratio (inter-class / intra-class distance).
        
        Args:
            avg_inter_class_distance: Average inter-class distance
            avg_intra_class_distance: Average intra-class distance
        
        Returns:
            Separation ratio
        """
        if avg_intra_class_distance == 0:
            return float('inf') if avg_inter_class_distance > 0 else 0.0
        return avg_inter_class_distance / avg_intra_class_distance
    
    def compute_silhouette_score(
        self, 
        embeddings: torch.Tensor, 
        labels: List[int]
    ) -> float:
        """
        Compute silhouette score for the clustering quality.
        
        Args:
            embeddings: Tensor of embeddings
            labels: List of corresponding labels
        
        Returns:
            Silhouette score
        """
        from sklearn.metrics import silhouette_score
        
        embeddings_np = embeddings.cpu().numpy()
        
        if len(set(labels)) < 2:
            return 0.0
        
        try:
            score = silhouette_score(embeddings_np, labels, metric=self.distance_metric)
            return float(score)
        except Exception as e:
            logging.warning(f"Could not compute silhouette score: {e}")
            return 0.0
    
    def analyze_data_quality(
        self, 
        texts: List[str], 
        labels: List[int],
        classifier,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of data quality using distance metrics.
        
        Args:
            texts: List of text strings
            labels: List of corresponding labels
            classifier: RoBERTaClassifier instance
            batch_size: Batch size for processing
        
        Returns:
            Dictionary containing comprehensive quality analysis
        """
        # Compute basic distance metrics
        distance_metrics = self.compute_distance_metrics(texts, labels, classifier, batch_size)
        
        # Compute additional metrics
        embeddings = self.compute_embeddings(texts, classifier, batch_size)
        silhouette_score = self.compute_silhouette_score(embeddings, labels)
        separation_ratio = self.compute_separation_ratio(
            distance_metrics["avg_inter_class_distance"],
            distance_metrics["avg_intra_class_distance"]
        )
        
        # Compute text length statistics
        text_lengths = [len(text.split()) for text in texts]
        
        # Quality assessment
        quality_score = self._compute_quality_score(
            distance_metrics["avg_inter_class_distance"],
            distance_metrics["avg_intra_class_distance"],
            silhouette_score,
            separation_ratio
        )
        
        return {
            **distance_metrics,
            "silhouette_score": silhouette_score,
            "separation_ratio": separation_ratio,
            "quality_score": quality_score,
            "text_length_stats": {
                "mean": float(np.mean(text_lengths)),
                "std": float(np.std(text_lengths)),
                "min": int(np.min(text_lengths)),
                "max": int(np.max(text_lengths))
            }
        }
    
    def _compute_quality_score(
        self, 
        inter_class_dist: float, 
        intra_class_dist: float, 
        silhouette_score: float, 
        separation_ratio: float
    ) -> float:
        """
        Compute an overall quality score based on multiple metrics.
        
        Args:
            inter_class_dist: Average inter-class distance
            intra_class_dist: Average intra-class distance
            silhouette_score: Silhouette score
            separation_ratio: Separation ratio
        
        Returns:
            Quality score between 0 and 1
        """
        # Normalize metrics to [0, 1] range
        # Higher inter-class distance is better
        inter_score = min(inter_class_dist / 2.0, 1.0)  # Assume max reasonable distance is 2.0
        
        # Lower intra-class distance is better
        intra_score = max(0, 1.0 - intra_class_dist / 1.0)  # Assume max reasonable distance is 1.0
        
        # Silhouette score is already in [-1, 1], normalize to [0, 1]
        silhouette_score_norm = (silhouette_score + 1) / 2
        
        # Separation ratio: higher is better, but cap at reasonable value
        separation_score = min(separation_ratio / 10.0, 1.0)  # Assume max reasonable ratio is 10.0
        
        # Weighted combination
        quality_score = (
            0.3 * inter_score +
            0.3 * intra_score +
            0.2 * silhouette_score_norm +
            0.2 * separation_score
        )
        
        return float(quality_score)
