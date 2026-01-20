"""
Evaluation utilities for aspect-based sentiment analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.aspect_sentiment_analyzer import AspectBasedSentimentAnalyzer, AspectSentimentResult
from src.data_processor import ReviewSample, DataProcessor


class AspectSentimentEvaluator:
    """Comprehensive evaluator for aspect-based sentiment analysis."""
    
    def __init__(self, analyzer: AspectBasedSentimentAnalyzer):
        """
        Initialize the evaluator.
        
        Args:
            analyzer: Trained aspect-based sentiment analyzer
        """
        self.analyzer = analyzer
    
    def evaluate_single_text(
        self, 
        text: str, 
        ground_truth: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluate analysis results for a single text.
        
        Args:
            text: Input text
            ground_truth: Dictionary mapping aspects to true sentiments
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = self.analyzer.analyze_text(text)
        
        # Convert predictions to dictionary
        pred_dict = {pred.aspect: pred.sentiment for pred in predictions}
        
        # Calculate metrics for each aspect
        results = {}
        for aspect in ground_truth.keys():
            if aspect in pred_dict:
                results[aspect] = {
                    'predicted': pred_dict[aspect],
                    'actual': ground_truth[aspect],
                    'correct': pred_dict[aspect] == ground_truth[aspect]
                }
            else:
                results[aspect] = {
                    'predicted': None,
                    'actual': ground_truth[aspect],
                    'correct': False
                }
        
        return results
    
    def evaluate_dataset(
        self, 
        dataset: List[ReviewSample]
    ) -> Dict[str, Any]:
        """
        Evaluate the analyzer on a complete dataset.
        
        Args:
            dataset: List of ReviewSample objects with ground truth
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        all_predictions = []
        all_ground_truth = []
        aspect_results = {}
        
        for sample in dataset:
            # Get predictions
            predictions = self.analyzer.analyze_text(sample.text)
            pred_dict = {pred.aspect: pred.sentiment for pred in predictions}
            
            # Compare with ground truth
            for aspect, true_sentiment in sample.aspect_sentiments.items():
                predicted_sentiment = pred_dict.get(aspect, None)
                
                all_predictions.append(predicted_sentiment)
                all_ground_truth.append(true_sentiment)
                
                if aspect not in aspect_results:
                    aspect_results[aspect] = {'predicted': [], 'actual': []}
                
                aspect_results[aspect]['predicted'].append(predicted_sentiment)
                aspect_results[aspect]['actual'].append(true_sentiment)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(all_ground_truth, all_predictions)
        
        # Calculate per-aspect metrics
        per_aspect_metrics = {}
        for aspect, results in aspect_results.items():
            per_aspect_metrics[aspect] = self._calculate_metrics(
                results['actual'], results['predicted']
            )
        
        return {
            'overall': overall_metrics,
            'per_aspect': per_aspect_metrics,
            'total_samples': len(dataset)
        }
    
    def _calculate_metrics(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        # Filter out None predictions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        
        if not valid_indices:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'coverage': 0.0
            }
        
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_valid, y_pred_valid, average='weighted', zero_division=0
        )
        coverage = len(valid_indices) / len(y_true)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'coverage': coverage
        }
    
    def create_confusion_matrix(
        self, 
        y_true: List[str], 
        y_pred: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create and display confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save the plot
        """
        # Filter out None predictions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        if not valid_indices:
            print("No valid predictions to create confusion matrix.")
            return
        
        # Get unique labels
        labels = sorted(list(set(y_true_valid + y_pred_valid)))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(
        self, 
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_dataset
            save_path: Optional path to save the report
            
        Returns:
            Formatted report string
        """
        report = ["=" * 80]
        report.append("ASPECT-BASED SENTIMENT ANALYSIS EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        overall = evaluation_results['overall']
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"Accuracy: {overall['accuracy']:.3f}")
        report.append(f"Precision: {overall['precision']:.3f}")
        report.append(f"Recall: {overall['recall']:.3f}")
        report.append(f"F1 Score: {overall['f1_score']:.3f}")
        report.append(f"Coverage: {overall['coverage']:.3f}")
        report.append("")
        
        # Per-aspect metrics
        report.append("PER-ASPECT PERFORMANCE:")
        report.append("-" * 40)
        
        for aspect, metrics in evaluation_results['per_aspect'].items():
            report.append(f"{aspect.upper()}:")
            report.append(f"  Accuracy: {metrics['accuracy']:.3f}")
            report.append(f"  Precision: {metrics['precision']:.3f}")
            report.append(f"  Recall: {metrics['recall']:.3f}")
            report.append(f"  F1 Score: {metrics['f1_score']:.3f}")
            report.append(f"  Coverage: {metrics['coverage']:.3f}")
            report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total Samples: {evaluation_results['total_samples']}")
        report.append(f"Number of Aspects: {len(evaluation_results['per_aspect'])}")
        
        # Best and worst performing aspects
        aspect_f1_scores = {
            aspect: metrics['f1_score'] 
            for aspect, metrics in evaluation_results['per_aspect'].items()
        }
        
        if aspect_f1_scores:
            best_aspect = max(aspect_f1_scores, key=aspect_f1_scores.get)
            worst_aspect = min(aspect_f1_scores, key=aspect_f1_scores.get)
            
            report.append(f"Best Performing Aspect: {best_aspect} (F1: {aspect_f1_scores[best_aspect]:.3f})")
            report.append(f"Worst Performing Aspect: {worst_aspect} (F1: {aspect_f1_scores[worst_aspect]:.3f})")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")
        
        return report_text


def run_evaluation_example():
    """Run a complete evaluation example."""
    # Generate synthetic dataset
    from src.data_processor import SyntheticDatasetGenerator
    
    generator = SyntheticDatasetGenerator()
    dataset = generator.generate_dataset(size=100)
    
    # Split into train and test
    train_dataset, test_dataset = DataProcessor.create_evaluation_dataset(dataset, test_size=0.3)
    
    # Initialize analyzer and evaluator
    analyzer = AspectBasedSentimentAnalyzer()
    evaluator = AspectSentimentEvaluator(analyzer)
    
    # Run evaluation
    print("Running evaluation on test dataset...")
    results = evaluator.evaluate_dataset(test_dataset)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results, "data/evaluation_report.txt")
    print(report)
    
    # Create visualizations
    evaluator.create_confusion_matrix(
        [sentiment for sample in test_dataset for sentiment in sample.aspect_sentiments.values()],
        [pred.sentiment for sample in test_dataset for pred in analyzer.analyze_text(sample.text)],
        "data/confusion_matrix.png"
    )


if __name__ == "__main__":
    run_evaluation_example()
