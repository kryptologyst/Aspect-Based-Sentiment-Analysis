#!/usr/bin/env python3
"""
Setup script for the Aspect-Based Sentiment Analysis project.

This script helps set up the project environment and download required models.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def setup_project():
    """Set up the project environment."""
    print("🚀 Setting up Aspect-Based Sentiment Analysis Project")
    print("=" * 60)
    
    # Create directories
    print("📁 Creating project directories...")
    directories = ["data", "models", "logs", "config", "tests", "web_app"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created {directory}/")
    
    # Create subdirectories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    print("  ✓ Created data subdirectories")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("⚠️  Failed to install requirements. Please install manually:")
        print("   pip install -r requirements.txt")
        return False
    
    # Download models (optional)
    print("🤖 Downloading pre-trained models...")
    try:
        from transformers import pipeline
        # Download sentiment analysis model
        pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        print("✅ Models downloaded successfully")
    except Exception as e:
        print(f"⚠️  Model download failed: {e}")
        print("   Models will be downloaded automatically on first use")
    
    # Generate sample data
    print("📊 Generating sample dataset...")
    try:
        from src.data_processor import SyntheticDatasetGenerator
        generator = SyntheticDatasetGenerator()
        dataset = generator.generate_dataset(size=100)
        generator.save_dataset(dataset, "data/sample_dataset.json")
        print("✅ Sample dataset generated")
    except Exception as e:
        print(f"⚠️  Sample dataset generation failed: {e}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the web app: python main.py web-app")
    print("2. Analyze text: python main.py analyze --text 'Your text here'")
    print("3. Generate dataset: python main.py generate-dataset --size 1000 --output data/my_dataset.json")
    print("4. Run tests: python -m pytest tests/ -v")


def check_requirements():
    """Check if requirements are met."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    else:
        print("✅ requirements.txt found")
    
    return True


def main():
    """Main setup function."""
    if not check_requirements():
        print("❌ Requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    setup_project()


if __name__ == "__main__":
    main()
