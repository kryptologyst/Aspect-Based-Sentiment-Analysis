"""
Aspect-Based Sentiment Analysis Module

This module provides comprehensive aspect-based sentiment analysis capabilities
using state-of-the-art transformer models and modern NLP techniques.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AspectSentimentResult:
    """Data class for storing aspect-based sentiment analysis results."""
    aspect: str
    sentiment: str
    confidence: float
    text_span: str
    start_pos: int
    end_pos: int


class AspectBasedSentimentAnalyzer:
    """
    Modern aspect-based sentiment analyzer using transformer models.
    
    This class provides comprehensive aspect-based sentiment analysis capabilities
    including aspect extraction, sentiment classification, and result visualization.
    """
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: Optional[str] = None
    ):
        """
        Initialize the aspect-based sentiment analyzer.
        
        Args:
            model_name: Hugging Face model name for sentiment analysis
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing AspectBasedSentimentAnalyzer with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Initialize the sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True
        )
        
        # Common aspect keywords for different domains
        self.aspect_keywords = {
            "food": ["food", "taste", "flavor", "delicious", "meal", "dish", "cuisine"],
            "service": ["service", "staff", "waiter", "waitress", "server", "helpful"],
            "ambiance": ["ambiance", "atmosphere", "environment", "decor", "music", "lighting"],
            "price": ["price", "cost", "expensive", "cheap", "affordable", "value", "worth"],
            "delivery": ["delivery", "shipping", "fast", "slow", "time", "arrived"],
            "quality": ["quality", "durable", "well-made", "poor", "excellent", "good"]
        }
    
    def extract_aspects(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract aspects from text using keyword matching and context analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of tuples containing (aspect, start_position, end_position)
        """
        aspects_found = []
        text_lower = text.lower()
        
        for aspect, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                # Find all occurrences of the keyword
                for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    start, end = match.span()
                    aspects_found.append((aspect, start, end))
        
        # Remove duplicates and sort by position
        aspects_found = list(set(aspects_found))
        aspects_found.sort(key=lambda x: x[1])
        
        return aspects_found
    
    def analyze_sentiment_for_aspect(
        self, 
        text: str, 
        aspect: str, 
        context_window: int = 50
    ) -> AspectSentimentResult:
        """
        Analyze sentiment for a specific aspect in the given text.
        
        Args:
            text: Input text
            aspect: Aspect to analyze
            context_window: Number of characters around aspect to consider
            
        Returns:
            AspectSentimentResult object with analysis results
        """
        # Find aspect mentions in text
        aspect_mentions = []
        text_lower = text.lower()
        
        for keyword in self.aspect_keywords.get(aspect, [aspect]):
            for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
                start, end = match.span()
                
                # Extract context around the aspect
                context_start = max(0, start - context_window)
                context_end = min(len(text), end + context_window)
                context_text = text[context_start:context_end]
                
                # Analyze sentiment of the context
                sentiment_result = self.sentiment_pipeline(context_text)
                
                # Get the highest confidence sentiment
                best_sentiment = max(sentiment_result[0], key=lambda x: x['score'])
                
                aspect_mentions.append({
                    'sentiment': best_sentiment['label'],
                    'confidence': best_sentiment['score'],
                    'text_span': context_text,
                    'start_pos': context_start,
                    'end_pos': context_end
                })
        
        if not aspect_mentions:
            # If no specific mentions found, analyze overall sentiment
            sentiment_result = self.sentiment_pipeline(text)
            best_sentiment = max(sentiment_result[0], key=lambda x: x['score'])
            
            return AspectSentimentResult(
                aspect=aspect,
                sentiment=best_sentiment['label'],
                confidence=best_sentiment['score'],
                text_span=text,
                start_pos=0,
                end_pos=len(text)
            )
        
        # Return the mention with highest confidence
        best_mention = max(aspect_mentions, key=lambda x: x['confidence'])
        
        return AspectSentimentResult(
            aspect=aspect,
            sentiment=best_mention['sentiment'],
            confidence=best_mention['confidence'],
            text_span=best_mention['text_span'],
            start_pos=best_mention['start_pos'],
            end_pos=best_mention['end_pos']
        )
    
    def analyze_text(
        self, 
        text: str, 
        aspects: Optional[List[str]] = None
    ) -> List[AspectSentimentResult]:
        """
        Perform comprehensive aspect-based sentiment analysis on the given text.
        
        Args:
            text: Input text to analyze
            aspects: Specific aspects to analyze (if None, auto-detect)
            
        Returns:
            List of AspectSentimentResult objects
        """
        logger.info(f"Analyzing text: {text[:100]}...")
        
        if aspects is None:
            # Auto-detect aspects
            detected_aspects = self.extract_aspects(text)
            aspects = list(set([aspect for aspect, _, _ in detected_aspects]))
        
        if not aspects:
            logger.warning("No aspects detected, analyzing overall sentiment")
            sentiment_result = self.sentiment_pipeline(text)
            best_sentiment = max(sentiment_result[0], key=lambda x: x['score'])
            
            return [AspectSentimentResult(
                aspect="overall",
                sentiment=best_sentiment['label'],
                confidence=best_sentiment['score'],
                text_span=text,
                start_pos=0,
                end_pos=len(text)
            )]
        
        results = []
        for aspect in aspects:
            result = self.analyze_sentiment_for_aspect(text, aspect)
            results.append(result)
        
        logger.info(f"Analysis complete. Found {len(results)} aspect sentiments.")
        return results
    
    def analyze_batch(
        self, 
        texts: List[str], 
        aspects: Optional[List[str]] = None
    ) -> List[List[AspectSentimentResult]]:
        """
        Analyze multiple texts for aspect-based sentiment.
        
        Args:
            texts: List of texts to analyze
            aspects: Specific aspects to analyze
            
        Returns:
            List of lists containing AspectSentimentResult objects
        """
        logger.info(f"Analyzing batch of {len(texts)} texts")
        
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            text_results = self.analyze_text(text, aspects)
            results.append(text_results)
        
        return results
    
    def visualize_results(
        self, 
        results: List[AspectSentimentResult], 
        save_path: Optional[str] = None
    ) -> None:
        """
        Create visualizations for aspect-based sentiment analysis results.
        
        Args:
            results: List of AspectSentimentResult objects
            save_path: Optional path to save the plot
        """
        if not results:
            logger.warning("No results to visualize")
            return
        
        # Prepare data for visualization
        aspects = [r.aspect for r in results]
        sentiments = [r.sentiment for r in results]
        confidences = [r.confidence for r in results]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sentiment distribution
        sentiment_counts = pd.Series(sentiments).value_counts()
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax1.set_title('Sentiment Distribution')
        
        # Confidence scores by aspect
        aspect_df = pd.DataFrame({
            'Aspect': aspects,
            'Confidence': confidences,
            'Sentiment': sentiments
        })
        
        sns.barplot(data=aspect_df, x='Aspect', y='Confidence', hue='Sentiment', ax=ax2)
        ax2.set_title('Confidence Scores by Aspect')
        ax2.set_ylabel('Confidence Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(
        self, 
        results: List[AspectSentimentResult]
    ) -> str:
        """
        Generate a comprehensive text report of the analysis results.
        
        Args:
            results: List of AspectSentimentResult objects
            
        Returns:
            Formatted report string
        """
        if not results:
            return "No analysis results available."
        
        report = ["=" * 60]
        report.append("ASPECT-BASED SENTIMENT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        aspects = [r.aspect for r in results]
        sentiments = [r.sentiment for r in results]
        confidences = [r.confidence for r in results]
        
        report.append(f"Total Aspects Analyzed: {len(results)}")
        report.append(f"Average Confidence: {np.mean(confidences):.3f}")
        report.append(f"Confidence Range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)
        
        for result in results:
            report.append(f"Aspect: {result.aspect}")
            report.append(f"  Sentiment: {result.sentiment}")
            report.append(f"  Confidence: {result.confidence:.3f}")
            report.append(f"  Text Span: {result.text_span[:100]}...")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function to demonstrate the aspect-based sentiment analyzer."""
    # Initialize the analyzer
    analyzer = AspectBasedSentimentAnalyzer()
    
    # Sample text for analysis
    sample_text = """
    The food at the restaurant was absolutely delicious and well-prepared. 
    However, the service was quite slow and the staff seemed overwhelmed. 
    The ambiance was cozy and romantic, perfect for a date night. 
    The prices were reasonable for the quality of food, though the portions 
    could have been larger. Overall, it was a good experience despite the service issues.
    """
    
    # Analyze the text
    print("Analyzing sample text...")
    results = analyzer.analyze_text(sample_text)
    
    # Generate and print report
    report = analyzer.generate_report(results)
    print(report)
    
    # Create visualization
    analyzer.visualize_results(results, save_path="data/aspect_sentiment_analysis.png")


if __name__ == "__main__":
    main()
