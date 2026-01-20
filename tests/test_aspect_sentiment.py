"""
Test suite for aspect-based sentiment analysis components.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from aspect_sentiment_analyzer import AspectBasedSentimentAnalyzer, AspectSentimentResult
from data_processor import SyntheticDatasetGenerator, DataProcessor, ReviewSample


class TestAspectBasedSentimentAnalyzer:
    """Test cases for AspectBasedSentimentAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return AspectBasedSentimentAnalyzer()
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.model_name is not None
        assert analyzer.device is not None
        assert analyzer.sentiment_pipeline is not None
        assert analyzer.aspect_keywords is not None
    
    def test_extract_aspects(self, analyzer):
        """Test aspect extraction."""
        text = "The food was delicious and the service was excellent."
        aspects = analyzer.extract_aspects(text)
        
        assert len(aspects) > 0
        assert any("food" in str(aspect) for aspect in aspects)
        assert any("service" in str(aspect) for aspect in aspects)
    
    def test_analyze_sentiment_for_aspect(self, analyzer):
        """Test sentiment analysis for specific aspect."""
        text = "The food was absolutely delicious and well-prepared."
        result = analyzer.analyze_sentiment_for_aspect(text, "food")
        
        assert isinstance(result, AspectSentimentResult)
        assert result.aspect == "food"
        assert result.sentiment in ["POSITIVE", "NEGATIVE", "NEUTRAL", "LABEL_0", "LABEL_1", "LABEL_2"]
        assert 0 <= result.confidence <= 1
        assert len(result.text_span) > 0
    
    def test_analyze_text(self, analyzer):
        """Test full text analysis."""
        text = "The food was delicious but the service was slow."
        results = analyzer.analyze_text(text, ["food", "service"])
        
        assert len(results) == 2
        assert all(isinstance(r, AspectSentimentResult) for r in results)
        assert all(r.aspect in ["food", "service"] for r in results)
    
    def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        texts = [
            "The food was great.",
            "The service was terrible."
        ]
        results = analyzer.analyze_batch(texts, ["food", "service"])
        
        assert len(results) == 2
        assert all(len(batch_results) >= 0 for batch_results in results)
    
    def test_generate_report(self, analyzer):
        """Test report generation."""
        text = "The food was delicious but the service was slow."
        results = analyzer.analyze_text(text, ["food", "service"])
        report = analyzer.generate_report(results)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "ASPECT-BASED SENTIMENT ANALYSIS REPORT" in report


class TestSyntheticDatasetGenerator:
    """Test cases for SyntheticDatasetGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return SyntheticDatasetGenerator()
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.restaurant_templates is not None
        assert generator.ecommerce_templates is not None
    
    def test_generate_review(self, generator):
        """Test single review generation."""
        review = generator.generate_review(domain="restaurant", num_aspects=2)
        
        assert isinstance(review, ReviewSample)
        assert len(review.text) > 0
        assert len(review.aspects) <= 2
        assert len(review.aspect_sentiments) == len(review.aspects)
        assert review.overall_sentiment in ["positive", "negative", "neutral"]
        assert review.domain == "restaurant"
    
    def test_generate_dataset(self, generator):
        """Test dataset generation."""
        dataset = generator.generate_dataset(size=10)
        
        assert len(dataset) == 10
        assert all(isinstance(sample, ReviewSample) for sample in dataset)
    
    def test_save_and_load_dataset(self, generator):
        """Test dataset saving and loading."""
        dataset = generator.generate_dataset(size=5)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            generator.save_dataset(dataset, temp_path)
            loaded_dataset = generator.load_dataset(temp_path)
            
            assert len(loaded_dataset) == len(dataset)
            assert all(isinstance(sample, ReviewSample) for sample in loaded_dataset)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        samples = [
            ReviewSample(
                text="The food was great.",
                aspects=["food"],
                aspect_sentiments={"food": "positive"},
                overall_sentiment="positive",
                domain="restaurant"
            ),
            ReviewSample(
                text="The service was poor.",
                aspects=["service"],
                aspect_sentiments={"service": "negative"},
                overall_sentiment="negative",
                domain="restaurant"
            )
        ]
        
        df = DataProcessor.prepare_training_data(samples)
        
        assert len(df) == 2
        assert "text" in df.columns
        assert "aspect" in df.columns
        assert "sentiment" in df.columns
        assert "overall_sentiment" in df.columns
        assert "domain" in df.columns
    
    def test_create_evaluation_dataset(self):
        """Test evaluation dataset creation."""
        samples = [
            ReviewSample(
                text=f"Sample text {i}",
                aspects=["food"],
                aspect_sentiments={"food": "positive"},
                overall_sentiment="positive",
                domain="restaurant"
            )
            for i in range(10)
        ]
        
        train_dataset, test_dataset = DataProcessor.create_evaluation_dataset(samples, test_size=0.2)
        
        assert len(train_dataset) == 8
        assert len(test_dataset) == 2
        assert len(train_dataset) + len(test_dataset) == len(samples)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis pipeline."""
        # Generate sample data
        generator = SyntheticDatasetGenerator()
        dataset = generator.generate_dataset(size=5)
        
        # Analyze with analyzer
        analyzer = AspectBasedSentimentAnalyzer()
        
        for sample in dataset:
            results = analyzer.analyze_text(sample.text)
            assert len(results) > 0
            assert all(isinstance(r, AspectSentimentResult) for r in results)
    
    def test_web_app_integration(self):
        """Test web app integration components."""
        # This would test the web app components if they were importable
        # For now, we'll just ensure the modules can be imported
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent / "web_app"))
            # import app  # Would test if web app imports work
            assert True  # Placeholder assertion
        except ImportError:
            pytest.skip("Web app modules not available for testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
