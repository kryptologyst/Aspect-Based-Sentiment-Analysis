#!/usr/bin/env python3
"""
Main entry point for the Aspect-Based Sentiment Analysis project.

This script provides a command-line interface for running various
aspects of the aspect-based sentiment analysis system.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from aspect_sentiment_analyzer import AspectBasedSentimentAnalyzer
from data_processor import SyntheticDatasetGenerator, DataProcessor
from evaluator import AspectSentimentEvaluator
from config_manager import get_config


def setup_logging():
    """Setup logging configuration."""
    config = get_config()
    
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def analyze_text_cli(text: str, aspects: list[str] = None):
    """Analyze text using command line interface."""
    analyzer = AspectBasedSentimentAnalyzer()
    
    print(f"Analyzing text: {text[:100]}...")
    results = analyzer.analyze_text(text, aspects)
    
    # Print results
    print("\n" + "="*60)
    print("ASPECT-BASED SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    for result in results:
        print(f"\nAspect: {result.aspect}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Text Span: {result.text_span}")
    
    # Generate report
    report = analyzer.generate_report(results)
    print(f"\n{report}")


def generate_dataset_cli(size: int, output_path: str):
    """Generate synthetic dataset using command line interface."""
    generator = SyntheticDatasetGenerator()
    
    print(f"Generating {size} synthetic samples...")
    dataset = generator.generate_dataset(size=size)
    
    # Save dataset
    generator.save_dataset(dataset, output_path)
    print(f"Dataset saved to {output_path}")
    
    # Show sample
    if dataset:
        sample = dataset[0]
        print(f"\nSample review:")
        print(f"Text: {sample.text}")
        print(f"Aspects: {sample.aspects}")
        print(f"Aspect Sentiments: {sample.aspect_sentiments}")
        print(f"Overall Sentiment: {sample.overall_sentiment}")


def evaluate_model_cli(dataset_path: str):
    """Evaluate model performance using command line interface."""
    generator = SyntheticDatasetGenerator()
    analyzer = AspectBasedSentimentAnalyzer()
    evaluator = AspectSentimentEvaluator(analyzer)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = generator.load_dataset(dataset_path)
    
    # Split dataset
    train_dataset, test_dataset = DataProcessor.create_evaluation_dataset(dataset, test_size=0.2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate_dataset(test_dataset)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results, "data/evaluation_report.txt")
    print(report)


def run_web_app():
    """Run the Streamlit web application."""
    import subprocess
    
    app_path = Path(__file__).parent / "web_app" / "app.py"
    
    if not app_path.exists():
        print(f"Web app not found at {app_path}")
        return
    
    print("Starting Streamlit web application...")
    print("The app will open in your default web browser.")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path)
    ])


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Aspect-Based Sentiment Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single text
  python main.py analyze --text "The food was great but service was slow"
  
  # Analyze with specific aspects
  python main.py analyze --text "Great food, poor service" --aspects food,service
  
  # Generate synthetic dataset
  python main.py generate-dataset --size 1000 --output data/synthetic.json
  
  # Evaluate model
  python main.py evaluate --dataset data/synthetic.json
  
  # Run web application
  python main.py web-app
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze text for aspect-based sentiment')
    analyze_parser.add_argument('--text', required=True, help='Text to analyze')
    analyze_parser.add_argument('--aspects', help='Comma-separated list of aspects to analyze')
    
    # Generate dataset command
    generate_parser = subparsers.add_parser('generate-dataset', help='Generate synthetic dataset')
    generate_parser.add_argument('--size', type=int, default=1000, help='Number of samples to generate')
    generate_parser.add_argument('--output', required=True, help='Output file path')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    evaluate_parser.add_argument('--dataset', required=True, help='Path to dataset file')
    
    # Web app command
    web_parser = subparsers.add_parser('web-app', help='Run Streamlit web application')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging()
    
    # Execute command
    try:
        if args.command == 'analyze':
            aspects = args.aspects.split(',') if args.aspects else None
            analyze_text_cli(args.text, aspects)
        
        elif args.command == 'generate-dataset':
            generate_dataset_cli(args.size, args.output)
        
        elif args.command == 'evaluate':
            evaluate_model_cli(args.dataset)
        
        elif args.command == 'web-app':
            run_web_app()
    
    except Exception as e:
        logging.error(f"Error executing command: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
