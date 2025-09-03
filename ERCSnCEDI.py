"""
Emotion Recognition Confidence Score (ERCS) and Cross-Emotion Discrimination Index (CEDI)
Implementation for "Bridging Emotional Intelligence and Deep Learning" paper

Authors: [Your Names]
License: MIT
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from typing import Tuple, List, Dict, Optional, Union
import warnings


class EmotionMetrics:
    """
    Custom metrics for evaluating empathetic AI systems.
    
    This class implements:
    - ERCS: Emotion Recognition Confidence Score
    - CEDI: Cross-Emotion Discrimination Index
    """
    
    def __init__(self, num_classes: int = 7, emotion_names: Optional[List[str]] = None):
        """
        Initialize EmotionMetrics calculator.
        
        Args:
            num_classes: Number of emotion categories (default: 7 for FER2013)
            emotion_names: List of emotion names for better interpretability
        """
        self.num_classes = num_classes
        self.emotion_names = emotion_names or [
            'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
        ]
        
        # Define emotion pairs of interest for CEDI calculation
        self.emotion_pairs = [
            ('Fear', 'Surprise'),
            ('Angry', 'Sad'),
            ('Angry', 'Disgust'),
            ('Happy', 'Surprise'),
            ('Happy', 'Neutral'),
            ('Sad', 'Neutral'),
            ('Fear', 'Sad')
        ]
    
    def compute_ercs(self, 
                     predictions: np.ndarray, 
                     labels: np.ndarray,
                     return_components: bool = False) -> Union[float, Dict[str, float]]:
        """
        Compute Emotion Recognition Confidence Score (ERCS).
        
        ERCS measures how well-calibrated a model's confidence is when making predictions,
        particularly important for empathetic AI applications.
        
        Formula:
        ERCS = (1/N_correct) * Σ[max(σ(z_i)) * (1 - H(σ(z_i)))]
        
        Args:
            predictions: Softmax probabilities of shape (N, num_classes)
            labels: Ground truth labels of shape (N,)
            return_components: If True, return detailed components
            
        Returns:
            ERCS score (float) or dictionary with components if return_components=True
        """
        # Ensure inputs are numpy arrays
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
            
        # Get predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Find correctly classified samples
        correct_mask = (predicted_classes == labels)
        n_correct = np.sum(correct_mask)
        
        if n_correct == 0:
            warnings.warn("No correct predictions found. ERCS undefined.")
            return 0.0 if not return_components else {'ercs': 0.0, 'n_correct': 0}
        
        # Get probabilities for correct predictions
        correct_probs = predictions[correct_mask]
        
        # Calculate max probability (confidence) for each correct prediction
        max_probs = np.max(correct_probs, axis=1)
        
        # Calculate entropy for each correct prediction's probability distribution
        # Normalize by log(num_classes) to get entropy in [0, 1] range
        entropies = np.array([entropy(probs, base=self.num_classes) for probs in correct_probs])
        
        # Calculate ERCS components
        confidence_term = max_probs
        entropy_penalty = 1 - entropies  # Lower entropy = higher confidence quality
        
        # Compute final ERCS
        ercs_components = confidence_term * entropy_penalty
        ercs = np.mean(ercs_components)
        
        if return_components:
            return {
                'ercs': ercs,
                'mean_confidence': np.mean(max_probs),
                'mean_entropy': np.mean(entropies),
                'mean_entropy_penalty': np.mean(entropy_penalty),
                'n_correct': n_correct,
                'n_total': len(labels),
                'accuracy': n_correct / len(labels)
            }
        
        return ercs
    
    def compute_cedi(self, 
                     y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     emotion_pair: Optional[Tuple[str, str]] = None) -> Union[float, Dict[Tuple[str, str], float]]:
        """
        Compute Cross-Emotion Discrimination Index (CEDI).
        
        CEDI measures the ability to distinguish between theoretically related 
        or commonly confused emotion pairs.
        
        Formula:
        CEDI(i,j) = 1 - [(C(i→j) + C(j→i)) / (N(i) + N(j))]
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            emotion_pair: Specific emotion pair to compute CEDI for.
                         If None, compute for all predefined pairs.
            
        Returns:
            CEDI score (float) for specific pair or dictionary of all pairs
        """
        # Ensure inputs are numpy arrays
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # Build confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        def calculate_cedi_for_pair(i: int, j: int) -> float:
            """Calculate CEDI for a specific emotion index pair."""
            # Number of samples in each class
            n_i = np.sum(y_true == i)
            n_j = np.sum(y_true == j)
            
            # Avoid division by zero
            if n_i + n_j == 0:
                return np.nan
            
            # Confusion counts
            c_i_to_j = cm[i, j]  # Class i misclassified as j
            c_j_to_i = cm[j, i]  # Class j misclassified as i
            
            # Calculate CEDI
            cedi = 1 - (c_i_to_j + c_j_to_i) / (n_i + n_j)
            
            return cedi
        
        # If specific pair requested
        if emotion_pair is not None:
            if emotion_pair[0] not in self.emotion_names or emotion_pair[1] not in self.emotion_names:
                raise ValueError(f"Invalid emotion pair: {emotion_pair}")
            
            i = self.emotion_names.index(emotion_pair[0])
            j = self.emotion_names.index(emotion_pair[1])
            return calculate_cedi_for_pair(i, j)
        
        # Otherwise, compute for all predefined pairs
        cedi_scores = {}
        for pair in self.emotion_pairs:
            i = self.emotion_names.index(pair[0])
            j = self.emotion_names.index(pair[1])
            cedi_scores[pair] = calculate_cedi_for_pair(i, j)
        
        return cedi_scores
    
    def compute_all_metrics(self,
                            predictions: np.ndarray,
                            labels: np.ndarray,
                            verbose: bool = True) -> Dict:
        """
        Compute all empathetic AI metrics.
        
        Args:
            predictions: Softmax probabilities of shape (N, num_classes)
            labels: Ground truth labels of shape (N,)
            verbose: If True, print detailed results
            
        Returns:
            Dictionary containing all metrics
        """
        # Get predicted classes
        y_pred = np.argmax(predictions, axis=1)
        
        # Compute ERCS
        ercs_results = self.compute_ercs(predictions, labels, return_components=True)
        
        # Compute CEDI for all pairs
        cedi_results = self.compute_cedi(labels, y_pred)
        
        # Calculate average CEDI
        valid_cedi = [v for v in cedi_results.values() if not np.isnan(v)]
        avg_cedi = np.mean(valid_cedi) if valid_cedi else np.nan
        
        results = {
            'ercs': ercs_results['ercs'],
            'ercs_components': ercs_results,
            'cedi_pairs': cedi_results,
            'avg_cedi': avg_cedi
        }
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Pretty print the results."""
        print("\n" + "="*60)
        print("EMPATHETIC AI METRICS EVALUATION")
        print("="*60)
        
        # ERCS Results
        print("\n1. Emotion Recognition Confidence Score (ERCS)")
        print("-"*40)
        print(f"   ERCS Score: {results['ercs']:.4f}")
        print(f"   Accuracy: {results['ercs_components']['accuracy']:.4f}")
        print(f"   Mean Confidence: {results['ercs_components']['mean_confidence']:.4f}")
        print(f"   Mean Entropy Penalty: {results['ercs_components']['mean_entropy_penalty']:.4f}")
        
        # CEDI Results
        print("\n2. Cross-Emotion Discrimination Index (CEDI)")
        print("-"*40)
        for pair, score in results['cedi_pairs'].items():
            if not np.isnan(score):
                print(f"   {pair[0]:8s} - {pair[1]:8s}: {score:.4f}")
        print(f"\n   Average CEDI: {results['avg_cedi']:.4f}")
        print("="*60 + "\n")


class ERCSCalibration:
    """Extended calibration metrics for ERCS analysis."""
    
    @staticmethod
    def expected_calibration_error(predictions: np.ndarray, 
                                  labels: np.ndarray, 
                                  n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            predictions: Softmax probabilities
            labels: Ground truth labels
            n_bins: Number of bins for calibration
            
        Returns:
            ECE score (lower is better)
        """
        confidences = np.max(predictions, axis=1)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracies = (predicted_classes == labels)
        
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            n_bin = np.sum(bin_mask)
            
            if n_bin > 0:
                bin_accuracy = np.mean(accuracies[bin_mask])
                bin_confidence = np.mean(confidences[bin_mask])
                ece += (n_bin / len(labels)) * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    @staticmethod
    def maximum_calibration_error(predictions: np.ndarray,
                                 labels: np.ndarray,
                                 n_bins: int = 10) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        Args:
            predictions: Softmax probabilities
            labels: Ground truth labels
            n_bins: Number of bins for calibration
            
        Returns:
            MCE score (lower is better)
        """
        confidences = np.max(predictions, axis=1)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracies = (predicted_classes == labels)
        
        mce = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            n_bin = np.sum(bin_mask)
            
            if n_bin > 0:
                bin_accuracy = np.mean(accuracies[bin_mask])
                bin_confidence = np.mean(confidences[bin_mask])
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))
        
        return mce


# Example usage and testing
if __name__ == "__main__":
    # Simulate model predictions for testing
    np.random.seed(42)
    n_samples = 1000
    n_classes = 7
    
    # Generate synthetic predictions (softmax probabilities)
    logits = np.random.randn(n_samples, n_classes) * 2
    predictions = F.softmax(torch.tensor(logits), dim=1).numpy()
    
    # Generate synthetic labels with some correlation to predictions
    labels = np.argmax(predictions, axis=1)
    # Add some noise to make it more realistic
    noise_mask = np.random.random(n_samples) < 0.3  # 30% error rate
    labels[noise_mask] = np.random.randint(0, n_classes, np.sum(noise_mask))
    
    # Initialize metrics calculator
    metrics = EmotionMetrics(num_classes=n_classes)
    
    # Compute all metrics
    results = metrics.compute_all_metrics(predictions, labels, verbose=True)
    
    # Additional calibration metrics
    calibration = ERCSCalibration()
    ece = calibration.expected_calibration_error(predictions, labels)
    mce = calibration.maximum_calibration_error(predictions, labels)
    
    print(f"\nAdditional Calibration Metrics:")
    print(f"   Expected Calibration Error (ECE): {ece:.4f}")
    print(f"   Maximum Calibration Error (MCE): {mce:.4f}")