"""
Utility functions and helpers for the aspect-based sentiment analysis project.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def setup_directories(base_path: str = ".") -> None:
    """Create necessary directories for the project."""
    directories = [
        "data",
        "models", 
        "logs",
        "config",
        "tests",
        "web_app"
    ]
    
    for directory in directories:
        Path(base_path, directory).mkdir(exist_ok=True)
    
    # Create subdirectories
    Path(base_path, "data", "raw").mkdir(parents=True, exist_ok=True)
    Path(base_path, "data", "processed").mkdir(parents=True, exist_ok=True)


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        return {}


def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to JSON file with error handling."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON file {file_path}: {e}")
        return False


def format_confidence(confidence: float) -> str:
    """Format confidence score for display."""
    return f"{confidence:.3f}"


def format_sentiment(sentiment: str) -> str:
    """Format sentiment label for display."""
    sentiment_mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral", 
        "LABEL_2": "Positive",
        "NEGATIVE": "Negative",
        "NEUTRAL": "Neutral",
        "POSITIVE": "Positive"
    }
    return sentiment_mapping.get(sentiment, sentiment)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "median": np.median(values)
    }


def create_sample_data() -> List[Dict[str, Any]]:
    """Create sample data for demonstration purposes."""
    return [
        {
            "text": "The food was absolutely delicious and well-prepared. However, the service was quite slow.",
            "aspects": ["food", "service"],
            "expected_sentiments": {"food": "positive", "service": "negative"}
        },
        {
            "text": "Great ambiance and reasonable prices, but the food was mediocre.",
            "aspects": ["ambiance", "price", "food"],
            "expected_sentiments": {"ambiance": "positive", "price": "positive", "food": "negative"}
        },
        {
            "text": "Excellent service and delicious food, though it's quite expensive.",
            "aspects": ["service", "food", "price"],
            "expected_sentiments": {"service": "positive", "food": "positive", "price": "negative"}
        }
    ]


def validate_text_input(text: str) -> bool:
    """Validate text input for analysis."""
    if not text or not text.strip():
        return False
    
    if len(text.strip()) < 10:
        return False
    
    return True


def validate_aspects(aspects: List[str]) -> bool:
    """Validate aspect list input."""
    if not aspects:
        return False
    
    if not all(isinstance(aspect, str) and aspect.strip() for aspect in aspects):
        return False
    
    return True


def get_model_info(model_name: str) -> Dict[str, str]:
    """Get information about a model."""
    model_info = {
        "cardiffnlp/twitter-roberta-base-sentiment-latest": {
            "description": "RoBERTa model fine-tuned on Twitter data",
            "max_length": "512",
            "languages": "English",
            "task": "Sentiment Analysis"
        },
        "bert-base-uncased": {
            "description": "BERT base model (uncased)",
            "max_length": "512", 
            "languages": "English",
            "task": "General NLP"
        },
        "distilbert-base-uncased-finetuned-sst-2-english": {
            "description": "DistilBERT fine-tuned on SST-2",
            "max_length": "512",
            "languages": "English", 
            "task": "Sentiment Analysis"
        }
    }
    
    return model_info.get(model_name, {
        "description": "Custom model",
        "max_length": "512",
        "languages": "Unknown",
        "task": "Unknown"
    })


def export_results_to_csv(results: List[Any], output_path: str) -> bool:
    """Export analysis results to CSV file."""
    try:
        # Convert results to DataFrame format
        data = []
        for result in results:
            if hasattr(result, '__dict__'):
                data.append(result.__dict__)
            else:
                data.append(result)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        logging.error(f"Error exporting to CSV {output_path}: {e}")
        return False


def create_progress_bar(total: int, current: int) -> str:
    """Create a simple text progress bar."""
    if total == 0:
        return "[                    ] 0%"
    
    progress = current / total
    filled = int(progress * 20)
    bar = "█" * filled + "░" * (20 - filled)
    percentage = int(progress * 100)
    
    return f"[{bar}] {percentage}%"


def log_performance_metrics(metrics: Dict[str, float], logger: logging.Logger) -> None:
    """Log performance metrics in a formatted way."""
    logger.info("Performance Metrics:")
    logger.info("-" * 30)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.3f}")
        else:
            logger.info(f"{metric}: {value}")
    
    logger.info("-" * 30)


def check_system_requirements() -> Dict[str, bool]:
    """Check if system meets requirements."""
    requirements = {
        "python_version": False,
        "torch_available": False,
        "transformers_available": False,
        "cuda_available": False
    }
    
    # Check Python version
    import sys
    requirements["python_version"] = sys.version_info >= (3, 8)
    
    # Check PyTorch
    try:
        import torch
        requirements["torch_available"] = True
        requirements["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Check Transformers
    try:
        import transformers
        requirements["transformers_available"] = True
    except ImportError:
        pass
    
    return requirements


def print_system_info() -> None:
    """Print system information."""
    requirements = check_system_requirements()
    
    print("System Requirements Check:")
    print("=" * 30)
    
    for req, status in requirements.items():
        status_str = "✓" if status else "✗"
        print(f"{req}: {status_str}")
    
    print("=" * 30)


if __name__ == "__main__":
    print_system_info()
