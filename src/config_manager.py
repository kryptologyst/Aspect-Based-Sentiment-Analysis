"""
Configuration management for the aspect-based sentiment analysis project.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str
    device: str
    max_length: int
    batch_size: int


@dataclass
class AspectConfig:
    """Aspect configuration parameters."""
    restaurant: list[str]
    ecommerce: list[str]
    general: list[str]


@dataclass
class AnalysisConfig:
    """Analysis configuration parameters."""
    context_window: int
    confidence_threshold: float
    auto_detect_aspects: bool


@dataclass
class DatasetConfig:
    """Dataset configuration parameters."""
    synthetic_size: int
    domains: list[str]
    sentiment_distribution: dict[str, float]


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str
    format: str
    file: str


@dataclass
class WebAppConfig:
    """Web application configuration parameters."""
    title: str
    page_icon: str
    layout: str
    theme: str


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config_data = self._load_config()
        
        # Initialize configuration objects
        self.model = ModelConfig(**self._config_data['model'])
        self.aspects = AspectConfig(**self._config_data['aspects'])
        self.analysis = AnalysisConfig(**self._config_data['analysis'])
        self.dataset = DatasetConfig(**self._config_data['dataset'])
        self.logging = LoggingConfig(**self._config_data['logging'])
        self.web_app = WebAppConfig(**self._config_data['web_app'])
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        if output_path is None:
            output_path = self.config_path
        
        config_dict = {
            'model': {
                'name': self.model.name,
                'device': self.model.device,
                'max_length': self.model.max_length,
                'batch_size': self.model.batch_size
            },
            'aspects': {
                'restaurant': self.aspects.restaurant,
                'ecommerce': self.aspects.ecommerce,
                'general': self.aspects.general
            },
            'analysis': {
                'context_window': self.analysis.context_window,
                'confidence_threshold': self.analysis.confidence_threshold,
                'auto_detect_aspects': self.analysis.auto_detect_aspects
            },
            'dataset': {
                'synthetic_size': self.dataset.synthetic_size,
                'domains': self.dataset.domains,
                'sentiment_distribution': self.dataset.sentiment_distribution
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file': self.logging.file
            },
            'web_app': {
                'title': self.web_app.title,
                'page_icon': self.web_app.page_icon,
                'layout': self.web_app.layout,
                'theme': self.web_app.theme
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_aspects_for_domain(self, domain: str) -> list[str]:
        """Get aspects for a specific domain."""
        if hasattr(self.aspects, domain):
            return getattr(self.aspects, domain)
        return self.aspects.general
    
    def update_model_config(self, **kwargs) -> None:
        """Update model configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
    
    def update_analysis_config(self, **kwargs) -> None:
        """Update analysis configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.analysis, key):
                setattr(self.analysis, key, value)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global config
    config = Config(config_path)
    return config
