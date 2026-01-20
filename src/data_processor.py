"""
Data processing and synthetic dataset generation for aspect-based sentiment analysis.
"""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class ReviewSample:
    """Data class for storing review samples."""
    text: str
    aspects: List[str]
    aspect_sentiments: Dict[str, str]
    overall_sentiment: str
    domain: str


class SyntheticDatasetGenerator:
    """Generate synthetic datasets for aspect-based sentiment analysis."""
    
    def __init__(self):
        """Initialize the synthetic dataset generator."""
        self.restaurant_templates = {
            "food": {
                "positive": [
                    "The food was absolutely delicious and well-prepared.",
                    "The flavors were amazing and perfectly balanced.",
                    "Every dish was cooked to perfection.",
                    "The ingredients were fresh and high-quality.",
                    "The presentation was beautiful and appetizing."
                ],
                "negative": [
                    "The food was bland and tasteless.",
                    "The dishes were overcooked and dry.",
                    "The ingredients seemed stale and poor quality.",
                    "The presentation was sloppy and unappetizing.",
                    "The flavors were completely off."
                ],
                "neutral": [
                    "The food was okay, nothing special.",
                    "The dishes were average in taste.",
                    "The ingredients were decent quality.",
                    "The presentation was standard.",
                    "The flavors were acceptable."
                ]
            },
            "service": {
                "positive": [
                    "The service was excellent and attentive.",
                    "The staff was friendly and helpful.",
                    "Our server was knowledgeable about the menu.",
                    "The service was prompt and efficient.",
                    "The staff went above and beyond."
                ],
                "negative": [
                    "The service was terrible and slow.",
                    "The staff was rude and unhelpful.",
                    "Our server was clueless about the menu.",
                    "The service was extremely delayed.",
                    "The staff was completely inattentive."
                ],
                "neutral": [
                    "The service was adequate.",
                    "The staff was polite but not exceptional.",
                    "Our server was competent.",
                    "The service was reasonably fast.",
                    "The staff was professional."
                ]
            },
            "ambiance": {
                "positive": [
                    "The ambiance was cozy and romantic.",
                    "The atmosphere was perfect for a date night.",
                    "The decor was elegant and sophisticated.",
                    "The lighting was warm and inviting.",
                    "The music was pleasant and not too loud."
                ],
                "negative": [
                    "The ambiance was loud and chaotic.",
                    "The atmosphere was cold and unwelcoming.",
                    "The decor was outdated and tacky.",
                    "The lighting was too bright and harsh.",
                    "The music was too loud and annoying."
                ],
                "neutral": [
                    "The ambiance was standard.",
                    "The atmosphere was okay.",
                    "The decor was simple and clean.",
                    "The lighting was adequate.",
                    "The music was background noise."
                ]
            },
            "price": {
                "positive": [
                    "The prices were very reasonable for the quality.",
                    "Great value for money.",
                    "The portions were generous for the price.",
                    "Affordable prices for such good food.",
                    "Worth every penny."
                ],
                "negative": [
                    "The prices were way too expensive.",
                    "Poor value for money.",
                    "The portions were tiny for the price.",
                    "Overpriced for what you get.",
                    "Not worth the money at all."
                ],
                "neutral": [
                    "The prices were fair.",
                    "Decent value for money.",
                    "The portions were appropriate for the price.",
                    "Reasonable pricing.",
                    "Average value."
                ]
            }
        }
        
        self.ecommerce_templates = {
            "product_quality": {
                "positive": [
                    "The product quality is excellent and durable.",
                    "Well-made and high-quality materials.",
                    "The craftsmanship is outstanding.",
                    "Built to last and very sturdy.",
                    "Premium quality that exceeds expectations."
                ],
                "negative": [
                    "The product quality is poor and flimsy.",
                    "Cheap materials that break easily.",
                    "The craftsmanship is terrible.",
                    "Falls apart after minimal use.",
                    "Low quality that doesn't meet expectations."
                ],
                "neutral": [
                    "The product quality is decent.",
                    "Average materials and construction.",
                    "The craftsmanship is acceptable.",
                    "Reasonable quality for the price.",
                    "Standard quality that meets expectations."
                ]
            },
            "delivery": {
                "positive": [
                    "Fast and reliable delivery service.",
                    "Arrived earlier than expected.",
                    "Packaging was excellent and secure.",
                    "Delivery person was professional.",
                    "Tracking was accurate and helpful."
                ],
                "negative": [
                    "Slow and unreliable delivery service.",
                    "Arrived much later than expected.",
                    "Packaging was damaged and inadequate.",
                    "Delivery person was unprofessional.",
                    "Tracking was inaccurate and confusing."
                ],
                "neutral": [
                    "Standard delivery service.",
                    "Arrived on time as expected.",
                    "Packaging was adequate.",
                    "Delivery person was competent.",
                    "Tracking was functional."
                ]
            },
            "customer_service": {
                "positive": [
                    "Excellent customer service and support.",
                    "Helpful and responsive staff.",
                    "Quick resolution of issues.",
                    "Professional and friendly service.",
                    "Goes above and beyond to help."
                ],
                "negative": [
                    "Terrible customer service and support.",
                    "Unhelpful and unresponsive staff.",
                    "Slow resolution of issues.",
                    "Unprofessional and rude service.",
                    "Doesn't care about customer problems."
                ],
                "neutral": [
                    "Adequate customer service.",
                    "Competent staff.",
                    "Standard resolution times.",
                    "Professional service.",
                    "Meets basic expectations."
                ]
            },
            "price": {
                "positive": [
                    "Great value for the price.",
                    "Affordable and competitive pricing.",
                    "Worth every penny.",
                    "Excellent price-to-quality ratio.",
                    "Budget-friendly option."
                ],
                "negative": [
                    "Overpriced for what you get.",
                    "Poor value for money.",
                    "Not worth the high price.",
                    "Expensive compared to competitors.",
                    "Rip-off pricing."
                ],
                "neutral": [
                    "Fair pricing.",
                    "Reasonable value for money.",
                    "Average price for the quality.",
                    "Competitive with market rates.",
                    "Standard pricing."
                ]
            }
        }
    
    def generate_review(
        self, 
        domain: str = "restaurant", 
        num_aspects: int = 3,
        sentiment_distribution: Dict[str, float] = None
    ) -> ReviewSample:
        """
        Generate a synthetic review with aspect-based sentiments.
        
        Args:
            domain: Domain type ('restaurant' or 'ecommerce')
            num_aspects: Number of aspects to include
            sentiment_distribution: Distribution of sentiments
            
        Returns:
            ReviewSample object
        """
        if sentiment_distribution is None:
            sentiment_distribution = {"positive": 0.4, "negative": 0.3, "neutral": 0.3}
        
        templates = self.restaurant_templates if domain == "restaurant" else self.ecommerce_templates
        
        # Select random aspects
        available_aspects = list(templates.keys())
        selected_aspects = random.sample(available_aspects, min(num_aspects, len(available_aspects)))
        
        # Generate text components
        text_parts = []
        aspect_sentiments = {}
        
        for aspect in selected_aspects:
            # Select sentiment based on distribution
            sentiment = random.choices(
                list(sentiment_distribution.keys()),
                weights=list(sentiment_distribution.values())
            )[0]
            
            # Get template text
            template_text = random.choice(templates[aspect][sentiment])
            text_parts.append(template_text)
            aspect_sentiments[aspect] = sentiment
        
        # Combine text parts
        text = " ".join(text_parts)
        
        # Determine overall sentiment
        sentiment_scores = {"positive": 1, "neutral": 0, "negative": -1}
        overall_score = sum(sentiment_scores[sent] for sent in aspect_sentiments.values())
        
        if overall_score > 0:
            overall_sentiment = "positive"
        elif overall_score < 0:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        return ReviewSample(
            text=text,
            aspects=selected_aspects,
            aspect_sentiments=aspect_sentiments,
            overall_sentiment=overall_sentiment,
            domain=domain
        )
    
    def generate_dataset(
        self, 
        size: int = 1000,
        domains: List[str] = None,
        output_path: Optional[str] = None
    ) -> List[ReviewSample]:
        """
        Generate a synthetic dataset of reviews.
        
        Args:
            size: Number of samples to generate
            domains: List of domains to include
            output_path: Optional path to save the dataset
            
        Returns:
            List of ReviewSample objects
        """
        if domains is None:
            domains = ["restaurant", "ecommerce"]
        
        dataset = []
        
        for i in range(size):
            domain = random.choice(domains)
            num_aspects = random.randint(2, 4)
            
            # Vary sentiment distribution
            sentiment_distributions = [
                {"positive": 0.5, "negative": 0.2, "neutral": 0.3},  # Positive bias
                {"positive": 0.2, "negative": 0.5, "neutral": 0.3},  # Negative bias
                {"positive": 0.3, "negative": 0.3, "neutral": 0.4},  # Neutral bias
                {"positive": 0.4, "negative": 0.4, "neutral": 0.2},  # Balanced
            ]
            
            sentiment_dist = random.choice(sentiment_distributions)
            
            sample = self.generate_review(domain, num_aspects, sentiment_dist)
            dataset.append(sample)
        
        if output_path:
            self.save_dataset(dataset, output_path)
        
        return dataset
    
    def save_dataset(self, dataset: List[ReviewSample], output_path: str) -> None:
        """Save dataset to JSON file."""
        data = [asdict(sample) for sample in dataset]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_dataset(self, input_path: str) -> List[ReviewSample]:
        """Load dataset from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [ReviewSample(**sample) for sample in data]


class DataProcessor:
    """Utility class for processing and preparing data for analysis."""
    
    @staticmethod
    def prepare_training_data(dataset: List[ReviewSample]) -> pd.DataFrame:
        """
        Prepare dataset for training/evaluation.
        
        Args:
            dataset: List of ReviewSample objects
            
        Returns:
            DataFrame with processed data
        """
        data = []
        
        for sample in dataset:
            for aspect, sentiment in sample.aspect_sentiments.items():
                data.append({
                    'text': sample.text,
                    'aspect': aspect,
                    'sentiment': sentiment,
                    'overall_sentiment': sample.overall_sentiment,
                    'domain': sample.domain
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_evaluation_dataset(dataset: List[ReviewSample], test_size: float = 0.2):
        """
        Split dataset into train and test sets.
        
        Args:
            dataset: List of ReviewSample objects
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        random.shuffle(dataset)
        split_idx = int(len(dataset) * (1 - test_size))
        
        train_dataset = dataset[:split_idx]
        test_dataset = dataset[split_idx:]
        
        return train_dataset, test_dataset


def main():
    """Generate and save a synthetic dataset."""
    generator = SyntheticDatasetGenerator()
    
    print("Generating synthetic dataset...")
    dataset = generator.generate_dataset(size=500)
    
    # Save dataset
    output_path = "data/synthetic_reviews.json"
    generator.save_dataset(dataset, output_path)
    
    print(f"Generated {len(dataset)} samples")
    print(f"Dataset saved to {output_path}")
    
    # Show sample
    print("\nSample review:")
    sample = dataset[0]
    print(f"Text: {sample.text}")
    print(f"Aspects: {sample.aspects}")
    print(f"Aspect Sentiments: {sample.aspect_sentiments}")
    print(f"Overall Sentiment: {sample.overall_sentiment}")
    print(f"Domain: {sample.domain}")


if __name__ == "__main__":
    main()
